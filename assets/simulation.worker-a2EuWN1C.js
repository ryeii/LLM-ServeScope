(function(){"use strict";const i=new URL("/LLM-ServeScope/python/",self.location.origin);let a;async function n(s){const e=await fetch(new URL(s,i));if(!e.ok)throw new Error(`Could not load the bundled Python asset: ${s}`);return e}function c(){return a||(a=(async()=>{const{loadPyodide:s}=await import(new URL("pyodide.mjs",i).href),e=await s({indexURL:i.href});e.FS.mkdirTree("/servescope/backend"),e.FS.mkdirTree("/servescope/site-packages");const t=["__init__.py","catalog.py","calibration.py","gpu_functions.py","model.py"],o=await Promise.all(t.map(async r=>[r,await(await n(`backend/${r}`)).text()]));for(const[r,p]of o)e.FS.writeFile(`/servescope/backend/${r}`,p);return e.FS.writeFile("/servescope/backend/gpu-fit.json",await(await n("backend/gpu-fit.json")).text()),e.unpackArchive(await(await n("simpy-4.1.1-py3-none-any.whl")).arrayBuffer(),"zip",{extractDir:"/servescope/site-packages"}),await e.runPythonAsync(`
import sys, json
sys.path[:0] = ['/servescope', '/servescope/site-packages']
from backend.model import catalog, validate, simulate
`),e})(),a)}let l=Promise.resolve();self.onmessage=({data:s})=>{l=l.then(async()=>{try{if(s.path==="health"){self.postMessage({id:s.id,result:{status:"ready",behavioral:!0,native:{available:!1,local_only:!0,message:"Native LLMServingSim requires the local Python/ASTRA installation. Browser behavioral simulation is available."}}});return}if(!["catalog","validate","simulate"].includes(s.path))throw new Error("This operation requires the local LLMServingSim backend.");const e=await c();e.globals.set("_request_json",JSON.stringify({path:s.path,body:s.body}));const t=await e.runPythonAsync(`
_request = json.loads(_request_json)
if _request['path'] == 'catalog':
    _response = catalog()
elif _request['path'] == 'validate':
    _response = {'config': validate(_request['body'])}
else:
    _response = simulate(_request['body'])
    _response['provenance']['execution_runtime'] = 'Browser: Pyodide 314.0.7 / SimPy 4.1.1'
json.dumps(_response, allow_nan=False)
`);self.postMessage({id:s.id,result:JSON.parse(t)})}catch(e){const t=e instanceof Error?e.message:String(e),o=t.match(/ValueError: (.+)/);self.postMessage({id:s.id,error:o?o[1]:t})}})}})();
