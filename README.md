<div align="center">
<h1>RingID 🔍 </h1>
<h3>RingID: Rethinking Tree-Ring Watermarking for Enhanced Multi-Key Identification</h3>

[Hai Ci](https://scholar.google.com/citations?user=GMrjppAAAAAJ&hl=en)<sup>&#42;</sup>&nbsp; Pei Yang<sup>&#42;</sup>&nbsp; [Yiren Song](https://scholar.google.com/citations?user=L2YS0jgAAAAJ&hl=en&oi=ao)<sup>&#42;</sup>&nbsp; [Mike Zheng Shou](https://sites.google.com/view/showlab)

National University of Singapore

[Project Page](https://sites.google.com/view/ringid2?usp=sharing) | [Preprint](https://drive.google.com/file/d/1HJOKRzPsGAnFzLOR-WTaf2d7C0s9eTaG/view)

</div>

<img src="assets/teaser.png" width="1000">

**RingID** presents a robust diffusion image watermarking approach to imprint multiple keys. It bases on the training-free approach Tree-Ring<sub>[1]</sub>, but significantly enhances in both watermark verification and multi-key identification. 


## Motivation
<img src="assets/motivation.png" width="1000">
Tree-Ring demonstrates extraordinary robustness in watermark verification (detection). We are interested its power to distinguish between different keys.  We comprehensively evaluate Tree-Ring in identification and find that it doesn't have enough distinguishability to identify different keys. What's more, it is sensitive to various image transformations, and totally unable to cope with Rotation and Crop/Scale.  Further research shows that an overlooked operation "discarding the imaginary part" introduced in injection process empowers Tree-Ring extraordinary robustness, especially to rotation and cropping/scaling. However, it does not help identify different keys. This motivates us to rethink the limitations in Tree-Ring and devise stronger solutions.


## Method
<img src="assets/pipeline_v2.png" width="1000">
RingID identifies the limitations in Tree-Ring's design and suggests a series of approaches for enhanced distinguishability and robustness.


## Qualitative 
<img src="assets/qualitative.png" width="1000">

## Reference
[1] Wen, Yuxin, et al. "Tree-ring watermarks: Fingerprints for diffusion images that are invisible and robust." arXiv preprint arXiv:2305.20030 (2023).

## Updates
- Code will be released soon.
