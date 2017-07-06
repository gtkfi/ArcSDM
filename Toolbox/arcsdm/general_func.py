
import sys
import arcpy
import traceback
import config as cfg

PY2 = sys.version_info[0] == 2
PY34 = sys.version_info[0:2] >= (3, 4)

if PY2:
    from imp import reload

    def _reload_module(name):
        reload(name)
if PY34:
    import importlib

    def _reload_module(name):
        importlib.reload(name)

if cfg.VERBOSE:
    def verbose_print(text, messages):
        messages.AddMessage("Verbose: " + text)
else:
    verbose_print = lambda *a: None


def print_parameters(parameters, messages):
    for var, par in enumerate(parameters):
        verbose_print("parameters[" + str(var) + "](" + str(par.name) + "): " + str(par.valueAsText), messages)


def _reload_modules(messages, modules):
    arcsdm_modules = [m.__name__ for m in sys.modules.values() if m]
    intersect = list(set(arcsdm_modules) & set(modules))
    for m in intersect:
        try:
            _reload_module(sys.modules[m])
        except Exception as e:
            messages.AddMessage("Failed to reload module {}. Reason:{}".format(m, e.message))
        verbose_print("Reloaded {} module".format(m), messages)


def execute_tool(func, self, parameters, messages, modules=None):
    if cfg.RELOAD_MODULES:
        if modules is None:
            modules = [func.__module__]
        # reload arcsdm.* modules
        _reload_modules(messages, modules)
        # update func ref to use reloaded code
        func.__code__ = getattr(sys.modules[func.__module__],  func.__name__).__code__
    if cfg.USE_PTVS_DEBUGGER:
        messages.AddMessage("Waiting for debugger..")
        try:
            from debug_ptvs import wait_for_debugger
            wait_for_debugger()
        except:
            messages.AddMessage("Failed to import debug_ptvs. Is ptvsd package installed?")
            raise
    try:
        # run the tool
        func(self, parameters, messages)
    except:
        errors = "Unhandled exception caught\n" + traceback.format_exc()
        arcpy.AddError(errors)
        raise
