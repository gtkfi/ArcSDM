# Receiver Operator Characteristics Toolbox
#
# Kimmo Korhonen / GTK

import pylab
import arcpy
import numpy
import os




NUM_REALIZATIONS = 10000
DISPLAY_INTERVAL = 1000
    
def execute(self, parameters, messages):
    positives_param, negatives_param, models_param, table_param = parameters

    positives_descr = arcpy.Describe(positives_param.valueAsText)
    positives_x, positives_y = FetchCoordinates(positives_descr.catalogPath)

    negatives_descr = negatives_param.valueAsText and arcpy.Describe(negatives_param.valueAsText) or None

    max_rows = 0
    tokens = models_param.valueAsText.split(";")
    model_names, roc_curves, auc_values, auc_confints = [], [], [], []
    for i in range(len(tokens)):
        raster_descr = arcpy.Describe(tokens[i])

        _roc_curve, _roc_ints, _auc_value, _auc_confints = CalculateCurveForModel(messages, raster_descr, negatives_descr, positives_x, positives_y)

        messages.addMessage("%s: AUC = %.3f." % (raster_descr.name, _auc_value))
        if _auc_confints:
            messages.addMessage("%s: 95%% confidence interval = %.3f-%.3f." % (raster_descr.name, _auc_confints[0], _auc_confints[1]))

        max_rows = numpy.max([max_rows, len(_roc_curve)])
        model_names.append(raster_descr.name)
        roc_curves.append(_roc_curve)
        auc_values.append(_auc_value)
        auc_confints.append(_auc_confints)

    table_path, table_name = os.path.split(table_param.valueAsText)
    arcpy.CreateTable_management(table_path, table_name)
    arcpy.AddField_management(table_param.valueAsText, "MODEL", "TEXT", field_length=10)
    arcpy.AddField_management(table_param.valueAsText, "AUC", "TEXT", field_length=10)
    if not negatives_descr:
        arcpy.AddField_management(table_param.valueAsText, "AUC_LO", "TEXT", field_length=10)
        arcpy.AddField_management(table_param.valueAsText, "AUC_HI", "TEXT", field_length=10)
    for i in range(len(model_names)):
        arcpy.AddField_management(table_param.valueAsText, "FPR_%d" % (i + 1), "DOUBLE", 20, 10, field_length=10)
        arcpy.AddField_management(table_param.valueAsText, "TPR_%d" % (i + 1), "DOUBLE", 20, 10, field_length=10)
    arcpy.DeleteField_management(table_param.valueAsText, "Field1")

    cursor = arcpy.InsertCursor(table_param.valueAsText)
    for i in range(max_rows):
        row = cursor.newRow()
        if i < len(model_names):
            row.setValue("MODEL", model_names[i])
            row.setValue("AUC", "%.3f" % auc_values[i])
            if not negatives_descr:
                row.setValue("AUC_LO", "%.3f" % auc_confints[i][0])
                row.setValue("AUC_HI", "%.3f" % auc_confints[i][1])
        for j in range(len(model_names)):
            if len(roc_curves[j]) > i:
                row.setValue("FPR_%d" % (j + 1), roc_curves[j][i, 0])
                row.setValue("TPR_%d" % (j + 1), roc_curves[j][i, 1])
        cursor.insertRow(row)
    del cursor, row

    
def FetchCoordinates(path):
    features, coords = arcpy.FeatureSet(path), []
    for row in arcpy.da.SearchCursor(features, ["SHAPE@XY"]):
        coords.append(row[0])
    coords = numpy.array(coords)
    return coords[:, 0], coords[:, 1]   
    
def CalculateCurveForModel(messages, raster_descr, negatives_descr, positives_x, positives_y):
    model_raster = arcpy.sa.Raster(raster_descr.catalogPath)
    raster_sampler = RasterSampler(model_raster)

    positives_x, positives_y, positive_sample = raster_sampler.Sample(positives_x, positives_y)

    if negatives_descr:
        negatives_x, negatives_y = FetchCoordinates(negatives_descr.catalogPath)
        negatives_x, negatives_y, negative_sample = raster_sampler.Sample(negatives_x, negatives_y)

        roc_curve = CalculateROCCurve(positive_sample, negative_sample)
        auc_value = CalculateAUCValue(roc_curve)

        return roc_curve, None, auc_value, None

    else:
        messages.addMessage("Performing Monte Carlo simulation (%d realizations)." % self.NUM_REALIZATIONS)

        fp = numpy.linspace(0, 1, 10*len(positives_x))
        tp = numpy.zeros((self.NUM_REALIZATIONS, len(fp)))

        roc_curves, auc_values = [], []
        for i in range(self.NUM_REALIZATIONS):
            negatives_x, negatives_y, negative_sample = raster_sampler.GenerateRandomSample(len(positives_x))
            roc_curve = self.CalculateROCCurve(positive_sample, negative_sample)
            auc_value = self.CalculateAUCValue(roc_curve)

            roc_curves.append(roc_curve)
            auc_values.append(auc_value)

            if (i + 1) % self.DISPLAY_INTERVAL == 0:
                messages.addMessage("Realization #%d (AUC = %.3f)" % (i + 1, auc_value))

            tp[i, :] = numpy.interp(fp, roc_curve[:,0], roc_curve[:,1])

        # Vertical statistics
        sorted_aucs = sorted(auc_values)

        lower_index = int(numpy.ceil(0.025 * self.NUM_REALIZATIONS)) - 1
        upper_index = int(numpy.floor(0.975 * self.NUM_REALIZATIONS)) - 1
        center_index = int(self.NUM_REALIZATIONS/2)

        lower_auc, upper_auc = sorted_aucs[lower_index], sorted_aucs[upper_index]

        lower_roc, central_roc, upper_roc = [], [], []
        for i in range(len(fp)):
            sorted_tp = sorted(tp[:, i])
            lower_roc.append([fp[i], sorted_tp[lower_index]])
            central_roc.append([fp[i], sorted_tp[center_index]])
            upper_roc.append([fp[i], sorted_tp[upper_index]])

        central_auc = self.CalculateAUCValue(central_roc)

        messages.addMessage("Completed Monte Carlo simulation.")

        return numpy.array(central_roc), [numpy.array(upper_roc), numpy.array(lower_roc)], central_auc, [lower_auc, upper_auc]

def CalculateAUCValue( roc_curve):
    auc_value = 0
    for i in range(len(roc_curve)-1):
        fp1, tp1 = roc_curve[i + 0]
        fp2, tp2 = roc_curve[i + 1]
        triangle_base = abs(fp1 - fp2)
        average_height = (tp1 + tp2) / 2.0
        auc_value += triangle_base * average_height
    return auc_value

def CalculateROCCurve( positive_sample, negative_sample):
    num_positives, num_negatives = len(positive_sample), len(negative_sample)

    classifier_values = numpy.hstack((positive_sample, negative_sample))
    classifier_classes = numpy.hstack((numpy.ones(num_positives), numpy.zeros(num_negatives)))

    if num_negatives == 0:
        raise ValueError("There are no negatives.")
    elif num_positives == 0:
        raise ValueError("There are no positives.")

    temp = [(x, y) for x, y in zip(classifier_values, classifier_classes)]
    temp = sorted(temp, key=lambda xy: xy[0], reverse=True)

    classifier_values, classifier_classes = [xy[0] for xy in temp], [xy[1] for xy in temp]

    roc_curve = []

    num_fp, num_tp = 0, 0

    prev_point = None

    for i in range(len(classifier_values)):
        next_point = (num_fp / float(num_negatives), num_tp / float(num_positives))

        roc_curve.append(next_point)

        if classifier_classes[i] == 1:
            num_tp += 1
        elif classifier_classes[i] == 0:
            num_fp += 1
        else:
            raise ValueError("Class values must be either 0 or 1, not %d." % classifier_classes[i])

        prev_point = next_point

    next_point = (1., 1.)

    roc_curve.append(next_point)

    return numpy.array(roc_curve)

    
    
class RasterSampler:

    NO_DATA_VALUE = -32768

    def __init__(self, raster):
        if raster.bandCount > 1:
            raise ValueError("Raster contains more than one band.")

        self.raster = raster

        self.x0 = raster.extent.upperLeft.X
        self.y0 = raster.extent.upperLeft.Y

        self.x1 = raster.extent.lowerRight.X
        self.y1 = raster.extent.lowerRight.Y

        self.num_rows = raster.height
        self.num_cols = raster.width

        self.pixel_width = (self.x1 - self.x0) / raster.width
        self.pixel_height = (self.y1 - self.y0) / raster.height
        
        if raster.isInteger:
            self.pixel_values = arcpy.RasterToNumPyArray(raster, nodata_to_value=self.NO_DATA_VALUE)
        else:
            self.pixel_values = arcpy.RasterToNumPyArray(raster, nodata_to_value=numpy.nan)

    def Sample(self, x, y):
        cols = numpy.floor((x - self.x0) / self.pixel_width).astype(int)
        rows = numpy.floor((y - self.y0) / self.pixel_height).astype(int)

        if self.raster.isInteger:
            z = numpy.ones(len(x)) * self.NO_DATA_VALUE
        else:
            z = numpy.ones(len(x)) * numpy.nan

        i = numpy.nonzero((cols >= 0) & (rows >= 0) & (cols < self.num_cols) & (rows < self.num_rows))
        z[i] = self.pixel_values[rows[i], cols[i]]

        if self.raster.isInteger:
            x, y, z = x[z != self.NO_DATA_VALUE], y[z != self.NO_DATA_VALUE], z[z != self.NO_DATA_VALUE]
        else:
            x, y, z = x[~numpy.isnan(z)], y[~numpy.isnan(z)], z[~numpy.isnan(z)]

        return x, y, z

    def GenerateRandomSample(self, sample_size):
        coords = numpy.zeros((sample_size, 2))
        values = numpy.zeros(sample_size)

        sample_count = 0
        while sample_count < sample_size:
            samples_remaining = sample_size - sample_count

            x = self.x0 + numpy.random.rand(samples_remaining) * (self.x1 - self.x0)
            y = self.y0 + numpy.random.rand(samples_remaining) * (self.y1 - self.y0)

            x, y, z = self.Sample(x, y)

            coords[sample_count:sample_count+len(z), :] = numpy.vstack((x, y)).T
            values[sample_count:sample_count+len(z)] = z

            sample_count += len(z)

        return coords[:,0], coords[:,1], values
