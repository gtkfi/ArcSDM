import ptvsd

def wait_for_debugger(timeout=None):
    try:
        ptvsd.enable_attach(secret=None)   
    except:
        pass
    ptvsd.wait_for_attach(timeout)
    return
