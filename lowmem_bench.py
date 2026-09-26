from __future__ import annotations
import os, time, gc, resource
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from brian2 import (
    Hz, Network, NeuronGroup, PoissonGroup, SpikeMonitor, Synapses,
    ms, mV, prefs
)
from flybrain import neurons as N

DATA = Path(os.environ.get("FLYBRAIN_DATA", "/srv/flybrain/data"))
PATH_COMP = DATA / "Completeness_783.csv"
PATH_CON = DATA / "Connectivity_783.parquet"

prefs.codegen.target = "numpy"
try:
    prefs.core.default_float_dtype = np.float32
except Exception as e:
    print(f"LOWMEM float32-pref warning: {e}", flush=True)
try:
    prefs.core.default_integer_dtype = np.int32
except Exception as e:
    print(f"LOWMEM int32-pref warning: {e}", flush=True)

PARAMS = dict(
    v_0=-52 * mV,
    v_rst=-52 * mV,
    v_th=-45 * mV,
    t_mbr=20 * ms,
    tau=5 * ms,
    t_rfc=2.2 * ms,
    t_dly=1.8 * ms,
    w_syn=0.275 * mV,
    r_poi=150 * Hz,
    f_poi=250,
)
EQS = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc                            : second
"""

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

t0 = time.time()
df_comp = pd.read_csv(PATH_COMP, index_col=0)
n_neurons = len(df_comp)
flyid2i = {int(j): i for i, j in enumerate(df_comp.index)}
stim_idx = {}
for name, ids in N.STIMULI.items():
    stim_idx[name] = np.asarray([flyid2i[i] for i in ids if i in flyid2i], dtype=np.int32)
mn9_idx = np.asarray([flyid2i[i] for i in (N.MN9_L, N.MN9_R) if i in flyid2i], dtype=np.int32)
all_stim_idx = np.unique(np.concatenate([v for v in stim_idx.values() if len(v)])).astype(np.int32)
print(f"LOWMEM comp loaded neurons={n_neurons} sugar={len(stim_idx['sugar'])} mn9={mn9_idx.tolist()} rss={rss_mb():.1f}MB", flush=True)

neu = NeuronGroup(
    N=n_neurons,
    model=EQS,
    method="linear",
    threshold="v > v_th",
    reset="v = v_rst; g = 0 * mV",
    refractory="rfc",
    name="neurons",
    namespace=PARAMS,
)
neu.v = PARAMS["v_0"]
neu.g = 0 * mV
neu.rfc = PARAMS["t_rfc"]

syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=PARAMS["t_dly"], name="synapses")

pf = pq.ParquetFile(PATH_CON)
n_synapses = pf.metadata.num_rows
print(f"LOWMEM parquet synapses={n_synapses} row_groups={pf.num_row_groups} rss={rss_mb():.1f}MB", flush=True)

cols = ["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
loaded = 0
batch_no = 0
for batch in pf.iter_batches(batch_size=250_000, columns=cols, use_threads=False):
    batch_no += 1
    pre = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.int32)
    post = np.asarray(batch.column(1).to_numpy(zero_copy_only=False), dtype=np.int32)
    weight = np.asarray(batch.column(2).to_numpy(zero_copy_only=False), dtype=np.float32)
    n = len(pre)
    start = len(syn)
    syn.connect(i=pre, j=post)
    syn.w[start:start+n] = weight * PARAMS["w_syn"]
    loaded += n
    del batch, pre, post, weight
    gc.collect()
    if batch_no == 1 or batch_no % 10 == 0 or loaded == n_synapses:
        print(f"LOWMEM edges loaded={loaded}/{n_synapses} batch={batch_no} rss={rss_mb():.1f}MB", flush=True)

print(f"LOWMEM connectivity ready synapses={len(syn)} rss={rss_mb():.1f}MB", flush=True)

poi = PoissonGroup(len(all_stim_idx), rates=0 * Hz, name="poisson")
poi_syn = Synapses(
    poi, neu, on_pre="v += w_poi",
    namespace={"w_poi": PARAMS["w_syn"] * PARAMS["f_poi"]},
    name="poisson_syn",
)
poi_syn.connect(i=np.arange(len(all_stim_idx), dtype=np.int32), j=all_stim_idx)

mon = SpikeMonitor(neu, name="spikes")
net = Network(neu, syn, poi, poi_syn, mon)

idx = stim_idx["sugar"]
rate_arr = np.zeros(len(all_stim_idx), dtype=np.float32) * Hz
stim_pos = np.searchsorted(all_stim_idx, idx)
rate_arr[stim_pos] = PARAMS["r_poi"]
poi.rates = rate_arr
neu.rfc[idx] = 0 * ms

print(f"LOWMEM built in {time.time()-t0:.1f}s neurons={n_neurons} synapses={len(syn)} rss={rss_mb():.1f}MB", flush=True)

sim_t0 = time.time()
n_seen = 0
for k in range(10):
    net.run(10 * ms)
    i_all = np.asarray(mon.i[:], dtype=np.int32)
    new = i_all[n_seen:]
    n_seen = len(i_all)
    mn9_so_far = int(np.isin(i_all, mn9_idx).sum())
    print(f"LOWMEM t={(k+1)*10}ms new_spikes={len(new)} total_spikes={len(i_all)} mn9_so_far={mn9_so_far} rss={rss_mb():.1f}MB", flush=True)

wall = time.time() - sim_t0
spikes_i = np.asarray(mon.i[:], dtype=np.int32)
mn9_spikes = int(np.isin(spikes_i, mn9_idx).sum())
active = int(len(np.unique(spikes_i)))
print(f"LOWMEM_RESULT sugar: sim 100 ms in {wall:.3f} s wall | MN9 spikes={mn9_spikes} eat={mn9_spikes >= 1} active={active} total_spikes={len(spikes_i)} rss={rss_mb():.1f}MB", flush=True)
