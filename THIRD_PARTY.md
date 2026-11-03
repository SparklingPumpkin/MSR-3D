# Third-party inference components

Scene loading, Gaussian rendering, crop conventions and MasQCLIP inference primitives are adapted from [OpenSplat3D](https://github.com/VisualComputingInstitute/opensplat3d). Gaussian Splatting-derived components retain the original Inria/GRAPHDECO notices and [research/evaluation license](third_party/licenses/gaussian-splatting.txt).

The mask-aware encoder uses the original MasQCLIP checkpoint and OpenAI CLIP. Segmentation utilities use Segment Anything's bounding-box operation. Those components and their weights retain their respective ownership and license terms.

The bundled feature-channel Gaussian rasterizer includes its source and original license under `third_party/diff-gaussian-rasterization`. GLM headers retain their notice in `third_party/diff-gaussian-rasterization/third_party/glm/COPYING`.
