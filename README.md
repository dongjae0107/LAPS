<div align="center">
<h1 align="center">LAPS: Improving Incremental LiDAR Mapping using Active Pooling and Sampling for Neural Distance Fields</h1>

<a href="https://arxiv.org/abs/2605.15496" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/arXiv-2605.15496-b31b1b?logo=arxiv" alt="arXiv"></a>
<!-- <a href="" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Paper-IEEE%20Xplore-00629B?logo=ieee" alt="paper"></a> -->

<p>
  <span class="author"><a href="https://dongjae0107.github.io/">Dongjae Lee</a><sup>1</sup></span> ·
  <span class="author"><a href="https://scholar.google.com/citations?user=lh2KUKMAAAAJ&hl=en&oi=ao">Wooseong Yang</a><sup>1</sup></span> ·
  <span class="author"><a href="https://yifutao.github.io/">Yifu Tao</a><sup>2</sup></span> ·
  <span class="author"><a href="https://scholar.google.com/citations?user=BqV8LaoAAAAJ&hl=en">Maurice Fallon</a><sup>2</sup></span> ·
  <span class="author"><a href="https://scholar.google.com/citations?user=7yveufgAAAAJ&hl=en">Ayoung Kim</a><sup>1</sup></span>
</p>

<sup>1</sup>[RPM Robotics Lab, Seoul National University](https://rpm.snu.ac.kr/); <sup>2</sup>[Dynamic Robot Systems Group, University of Oxford](https://dynamic.robots.ox.ac.uk/)
</div>

## TL;DR
LAPS improves incremental neural LiDAR mapping by actively managing replay samples under a fixed memory and training budget.
It consists of two key components: **Reliability-based Active Pooling**, which retains reliable replay samples while reducing spatial imbalance, and **Uncertainty-guided Active Sampling**, which allocates online training samples based on uncertainty.

<p align="center">
  <img src="assets/pipeline.svg" width="99%">
</p>
<p align="center">
  <em>Overview of the LAPS pipeline.</em>
</p>

## Updates
- **[June, 2026]**: Code will be released in June.
- **[May, 2026]**: LAPS was accepted to IEEE Robotics and Automation Letters (RA-L)!

<!-- ## Citation
If you use use this code or find our work useful for your research, please consider citing:
```bibtex
@article{
}
``` -->