"""
    General Functions

    This module contains functions used for other modules 

    Authors: Irving Cabrera <irvcaza@gmail.com>
"""
import sys
import arcpy
import traceback
import arcsdm.config as cfg

PY2 = sys.version_info[0] == 2
PY34 = sys.version_info[0:2] >= (3, 4)

# Change the importing behaviour depending in the version of python
if PY2:
    from imp import reload

    def _reload_module(name):
        reload(name)
if PY34:
    import importlib

    def _reload_module(name):
        importlib.reload(name)

# TODO: This is repeated in all tools, better link them to this function
# Verbose prints to give more information about the execution *In development*
if cfg.VERBOSE:
    def verbose_print(text, messages):
        messages.AddMessage("Verbose: " + text)
else:
    verbose_print = lambda *a: None

#
def print_parameters(parameters, messages):
    """  ****IN DEVELOPMENT**** 
            print_parameters
                Prints the element in the parameters object. Needs to have verbosity activated
            :param parameters: Object with the attributes name and valueAsText
            :return: none 
        """
    for var, par in enumerate(parameters):
        verbose_print("parameters[" + str(var) + "](" + str(par.name) + "): " + str(par.valueAsText), messages)


def _reload_modules(messages, modules):
    """
        _reload_modules
            Reloads the necessary modules for the execution of the tool 
            
    :param messages: messages object to print in the console, must implement AddMessage 
    :param modules: list with modules to be updated 
    """
    arcsdm_modules = [m.__name__ for m in sys.modules.values() if m]
    intersect = list(set(arcsdm_modules) & set(modules))
    for m in intersect:
        try:
            _reload_module(sys.modules[m])
        except Exception as e:
            messages.AddMessage("Failed to reload module {}. Reason:{}".format(m, e.message))
        verbose_print("Reloaded {} module".format(m), messages)


def execute_tool(func, self, parameters, messages, modules=None):
    """
        execute_tool
            Executes the tool after reloading necessary modules and optionally activate the debugger
    
    :param func: Function to be executed  
    :param parameters: Parameters given to the function 
    :param messages: Messages object necessary for the execution of the tool
    :param modules: List of modules to be reloaded. If none then only the tool module is reloaded
    :return: 
    """
    if cfg.RELOAD_MODULES:
        if modules is None:
            modules = [func.__module__]
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
