# MEC MADDPG
Python code written for my thesis' simulations.

**Thesis**: *Optimization through Deep Reinforcement Learning for Secure Data Offloading in Wireless Networks*

Analysis of a mobile edge computing (MEC) system containing 1 edge server (ES), 1 eavesdropper and $k$ mobile users (MUs). Each user tries to complete his task before a specific time limit. To achieve this, we enable data offloading through wireless communication using Rate-Splitting Multiple Access (RSMA). The eavesdropper tries to decode the transmitted data. The goal of the system is to minimize the total energy consumption while maintaining security.
To this end we deploy a multi-agent DRL algorithm called MADDPG (Multi-Agent Deep Deterministic Policy Gradient) [[1]](#1).

* [constants.py](constants.py): contains most of the constants used in the simulations
* [environment.py](environment.py): simulates the MEC environment
* [models.py](models.py): actor and critic neural networks
* [maddpg.py](maddpg.py): class containing actors and critics
* [memory.py](memory.py): memory implementation
* [model_utils.py](model_utils.py): save and load implementations for system model (maddpg class)
* [run_maddpg.py](run_maddpg.py): train or evaluate system model
* [logs_plot.py](logs_plot.py), [bar_plot.py](bar_plot.py): produce graphs from plots

To run simply execute: *python3 run_maddpg.py*.


<a id="1">[1]</a> Chen, Z., Zhang, L., Pei, Y., Jiang, C. and Yin, L., 2021. NOMA-based multi-user mobile edge computation offloading via cooperative multi-agent deep reinforcement learning. IEEE Transactions on Cognitive Communications and Networking, 8(1), pp.350-364.