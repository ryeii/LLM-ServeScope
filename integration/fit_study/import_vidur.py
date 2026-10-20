"""Archive selected exact published Vidur files and their MIT license, pinned by SHA.
Usage: python -m integration.fit_study.import_vidur /path/to/vidur
"""
from pathlib import Path
import gzip,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
PIN='abae7f63aa857300f5cdc6f5e0d27860cd24721b'
MODELS=['meta-llama/Llama-2-7b-hf','meta-llama/Llama-2-70b-hf','codellama/CodeLlama-34b-Instruct-hf']
def main():
    source=Path(sys.argv[1]); dest=ROOT/'third_party/vidur-profiles'; dest.mkdir(exist_ok=True)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==PIN
    assert not subprocess.check_output(['git','diff','--ignore-submodules'],cwd=source,text=True)
    paths=['LICENSE','vidur/config/device_sku_config.py','vidur/config/model_config.py',
           'vidur/profiling/common/cuda_timer.py','vidur/profiling/common/timer_stats_store.py',
           'vidur/profiling/common/model_config.py','vidur/profiling/mlp/mlp_wrapper.py',
           'vidur/profiling/mlp/mlp_impl.py','vidur/profiling/attention/attention_wrapper.py',
           'vidur/profiling/utils/record_function_tracer.py']
    paths += [f'data/profiling/compute/{g}/{m}/{name}.csv' for g in ['a100','a40','h100'] for m in MODELS for name in ['mlp','attention']]
    files=[]
    for path in paths:
        raw=(source/path).read_bytes(); archive=path.replace('/','__')+'.gz'
        (dest/archive).write_bytes(gzip.compress(raw,mtime=0))
        files.append(dict(path=path,archive=archive,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),url=f'https://github.com/microsoft/vidur/blob/{PIN}/{path}'))
    manifest=dict(repository='https://github.com/microsoft/vidur',revision=PIN,license='MIT',files=files,
      timing_unit='milliseconds; CSV means converted to nanoseconds by multiplying 1e6, confirmed in CudaTimer and RecordFunctionTracer',
      dtype='FP16, from mlp_wrapper.py and attention/main.py; not pooled with KAIST BF16',
      software='Sarathi/vLLM-derived kernels; exact historical torch/CUDA/driver versions are not supplied in these CSVs. Current source is evidence of profiler contracts, not proof of each historical run environment.',
      hardware='Device variants inferred from Vidur SKU definitions (A100 80GB, H100 80GB, A40). Bandwidth/SM/vector peaks supplied as documented reference assumptions, with variant sensitivity audit.')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Archived',len(files),'files',sum((dest/f['archive']).stat().st_size for f in files),'compressed bytes')
if __name__=='__main__': main()
