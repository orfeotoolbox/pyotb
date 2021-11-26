import multiprocessing
from pyotb.core import App

def get_available_applications(q):
    import otbApplication
    q.put(otbApplication.Registry.GetAvailableApplications())

# We run this piece of code inside a independent `multiprocessing.Process` because of a current (2021-11) bug that
# prevents the use of OTBTF and tensorflow inside the same script
q = multiprocessing.Queue()
p = multiprocessing.Process(target=get_available_applications, args=(q,))
p.start()
p.join()
AVAILABLE_APPLICATIONS = q.get(block=False)

# This is to enable aliases of Apps, i.e. using apps like `pyotb.AppName(...)` instead of `pyotb.App('AppName', ...)`
if AVAILABLE_APPLICATIONS:
    for app_name in AVAILABLE_APPLICATIONS:
        exec(f"""def {app_name}(*args, **kwargs): return App('{app_name}', *args, **kwargs)""")