"""BNCI clean-data comparison. Run with .venv/bin/python experiments/benchmark_gedai_050_bnci.py.

Historical-compatible stratified 80/20 split over four classes; binary left/right
CSP+LDA scoring. No injection or epoch rejection. Continuous cleaning is offline,
per-run and label-blind, but uses the full run (transductive, not online CV).
"""
from pathlib import Path
import sys, json, time, argparse, hashlib, importlib.metadata
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.load_moabb import load_bnci2014_001, epochs_to_numpy
from datasets.preprocess import standard_preprocess, epochs_to_continuous, continuous_to_epochs
from metrics.decoding_metrics import build_csp_lda_pipeline
from gedai_core.gedai import GEDAICore
from gedai_core.leadfield import load_interpolated_leadfield
from gedai_core.reject import two_pass_channel_rejection
from gedai import Gedai, AdaptiveMultibandGedai
from moabb.datasets import BNCI2014_001
OUT=Path(__file__).resolve().parents[1]/'results'/'gedai_050_bnci'

def score(x,y,tr,te):
    tr=tr[np.isin(y[tr],[1,2])];te=te[np.isin(y[te],[1,2])]
    p=build_csp_lda_pipeline();p.fit(x[tr].astype(float),y[tr]);pred=p.predict(x[te].astype(float))
    return float(balanced_accuracy_score(y[te],pred)),pred.tolist(),te.tolist()

def broadband(x,info,tr,te):
    out=np.empty_like(x,dtype=float)
    # Optimize independently on each partition, as in historical cleaning.
    # fit_epochs avoids wavelets and artificial boundaries entirely.
    for idx in (tr,te):
        ep=mne.EpochsArray(x[idx]*1e-6,info,tmin=-.5,verbose=False)
        model=Gedai(engine='numpy')
        model.fit_epochs(ep,noise_multiplier=3.,n_jobs=1,verbose=False)
        out[idx]=model.transform_epochs(ep,n_jobs=1,verbose=False).get_data()*1e6
    return out

def continuous(subject, expected_epochs):
    ds=BNCI2014_001(); raws=ds.get_data(subjects=[subject])[subject]
    arrays=[];labels=[];keys=[];controls=[]
    for session,runs in raws.items():
        for run,raw in runs.items():
            print(f'S{subject} continuous {session}/{run}',flush=True)
            events=mne.find_events(raw,shortest_event=0,verbose=False)
            raw=raw.copy().pick('eeg').load_data()
            raw.filter(.5,100.,method="iir",verbose=False)
            control=mne.Epochs(raw,events,event_id=ds.event_id,tmin=1.5,tmax=6.,baseline=None,
                preload=True,proj=False,verbose=False)
            controls.append(control.get_data())
            model=AdaptiveMultibandGedai(engine='numpy')
            model.fit_raw(raw,noise_multiplier=3.,n_jobs=1,verbose=False)
            clean=model.transform_raw(raw,n_jobs=1,verbose=False)
            ep=mne.Epochs(clean,events,event_id=ds.event_id,tmin=1.5,tmax=6.,baseline=None,
                          preload=True,reject_by_annotation=False,verbose=False)
            assert len(ep)==len(events)
            arrays.append(ep.get_data());labels.extend(ep.events[:,2].tolist())
            keys.extend([[session,run,int(e[0])] for e in ep.events])
    control=np.concatenate(controls)
    np.testing.assert_allclose(control,expected_epochs,rtol=1e-6,atol=1e-12,
        err_msg="Raw extraction does not match MOABB baseline epochs")
    x=np.concatenate(arrays)*1e6
    return standard_preprocess(x,250.,apply_average_ref=False),np.array(labels),keys

def run(subject):
    t=time.time();sd=load_bnci2014_001(subjects=[subject]);x,y,names,sf=epochs_to_numpy(sd,subject)
    y=np.array([{'left_hand':1,'right_hand':2,'feet':3,'tongue':4}[str(v)] for v in y])
    info=sd[subject]['eeg_epochs'].info.copy()
    x=standard_preprocess(x,sf,apply_average_ref=False)
    tr,te=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
    results={'subject':subject,'n_epochs':len(y),'train_indices':tr.tolist(),'test_indices':te.tolist(),'methods':{}}
    def save(name,z):
        assert z.shape==x.shape and np.isfinite(z).all(),(name,z.shape)
        ba,pred,idx=score(z,y,tr,te)
        results['methods'][name]={'ba':ba,'predictions':pred,'scored_indices':idx,
            'relative_squared_change':float(np.sum((z-x)**2)/np.sum(x*x))}
        results['elapsed_seconds']=time.time()-t
        (OUT/f'subject_{subject}.json').write_text(json.dumps(results,indent=2))
        print(f'S{subject} {name}: {ba:.6f}',flush=True)
    save('bandpass_baseline',x)
    save('bandpass_average_reference',x-x.mean(axis=1,keepdims=True))
    positions=np.array([c['loc'][:3] for c in info['chs']])
    ref=load_interpolated_leadfield(names,positions)
    kwargs=dict(artifact_threshold_type='auto',epoch_size_in_cycles=12.,lowcut_hz=.5,smoothing_window_sec=np.inf)
    old=np.empty_like(x)
    old[tr]=continuous_to_epochs(GEDAICore(**kwargs).run(epochs_to_continuous(x[tr]),sf,names,ref_cov_override=ref).clean[:,:len(tr)*x.shape[2]],len(tr),x.shape[2])
    cleaned,_,_=two_pass_channel_rejection(data=epochs_to_continuous(x[te]),sfreq=sf,ch_names=names,enova_threshold_channel=.95,gedai_kwargs=kwargs,ref_cov=ref)
    old[te]=continuous_to_epochs(cleaned[:,:len(te)*x.shape[2]],len(te),x.shape[2])
    save('legacy_current_checkout',old)
    save('gedai_050_broadband_epochs',broadband(x,info,tr,te))
    z,yc,keys=continuous(subject,sd[subject]["eeg_epochs"].get_data())
    assert np.array_equal(y,yc),'Continuous and MOABB epoch label order differ'
    results['continuous_epoch_keys']=keys
    save('gedai_050_continuous',z)
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subjects',nargs='+',type=int,default=list(range(1,10)));args=p.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);mne.set_log_level('ERROR')
    import gedai;gedai.set_log_level('ERROR')
    versions={k:importlib.metadata.version(k) for k in ['gedai','moabb','mne','numpy','scipy','scikit-learn']}
    assert versions['gedai']=='0.5.0'
    (OUT/'source_hashes.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path('gedai_core/gedai.py'),Path('datasets/load_moabb.py'),Path('datasets/preprocess.py')]},indent=2))
    (OUT/'versions.json').write_text(json.dumps(versions,indent=2))
    rows=[]
    with threadpool_limits(limits=1):
        for s in args.subjects:
            r=run(s)
            rows.extend({'subject':s,'method':k,'ba':v['ba']} for k,v in r['methods'].items())
            pd.DataFrame(rows).to_csv(OUT/'scores.csv',index=False)
    print(pd.DataFrame(rows).groupby('method').ba.agg(['mean','std']).to_string(),flush=True)
