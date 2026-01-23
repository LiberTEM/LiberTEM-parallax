# libertem-parallax

**Streaming parallax imaging UDFs for 4D-STEM**, built on top of
[LiberTEM](https://github.com/LiberTEM/LiberTEM).

This package provides two streaming-capable user-defined functions (UDFs)
for parallax imaging. While applicable to offline analysis, the UDFs are
primarily designed for **live processing during acquisition** using
[LiberTEM-live](https://github.com/LiberTEM/LiberTEM-live).

## Installation

The package is available on the Python Package Index (PyPI) as
[`libertem-parallax`](https://pypi.org/project/libertem-parallax/).

Install using pip:

```bash
pip install libertem-parallax
```

For a developer installation, please refer to [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Usage example

The UDFs are not meant to be instantiated directly via `ParallaxUDF(...)`.
Instead, use the provided classmethod constructor to specify the acquisition and reconstruction parameters:

```python
import libertem.api
import libertem_parallax as prlx
import numpy as np

ctx = libertem.api.Context()

# dataset available at https://doi.org/10.5281/zenodo.18346853
ds = ctx.load("auto", path="../data/apoF_4mrad_1.5um-df_3A-step_30eA2_binary_uint8.npy")

udf = prlx.ParallaxUDF.from_parameters(
    shape=ds.shape,
    scan_sampling=(256/72,256/72),
    angular_sampling=(0.307617,0.307617),
    energy=300e3,
    semiangle_cutoff=4.0,
    aberration_coefs={"C10":-1.5e4},
    rotation_angle=np.deg2rad(-15),
    upsampling_factor=2,
)

udf_flip = prlx.ParallaxPhaseFlipUDF.from_parallax_udf(
    udf
)

result, result_flip = ctx.run_udf(
    dataset=ds, udf=[udf, udf_flip],
    progress=True, plots=True
)
```

## Parallax imaging background

Parallax imaging[^1], also known as tilt-corrected bright-field (tcBF) STEM[^2], is a STEM phase contrast technique which can be understood as a quadratic approximation[^3] to direct ptychography techniques such as phase-compensated single-sideband[^4].

In direct ptychography methods, phase contrast is recovered by applying a frequency-dependent deconvolution kernel to each diffraction pattern.
Parallax imaging replaces this with a simple yet remarkably robust bright-field–frequency-dependent phase ramp.
This leads to an important practical consequence:
> Parallax imaging replaces expensive Fourier-domain operations with simple, real-space updates.

Each bright-field pixel is associated with a real-space parallax shift, determined by the gradient of the aberration phase surface.
During acquisition, each diffraction frame contributes locally to the reconstruction at the shifted position.

Two closely related variants are implemented:

1. Parallax imaging (`ParallaxUDF`)    
    Each BF pixel contributes a **delta-like update** at its shifted location, correcting for the parallax effect.
2. Phase-flipped parallax imaging (`ParallaxPhaseFlipUDF`)  
     Each BF pixel contributes a shifted **fixed real-space kernel**, correcting for both the parallax effect **and contrast reversals**, enabling dark-field-like contrast.

## License

LiberTEM-parallax is licensed under the MIT license.

The project includes code adapted from existing [quantEM](https://github.com/electronmicroscopy/quantem) utilities where noted, under a compatible license.
See individual source files for attribution details.

[^1]: https://arxiv.org/abs/2309.05250
[^2]: https://www.nature.com/articles/s41592-025-02834-9
[^3]: https://arxiv.org/abs/2507.18610 
[^4]: https://doi.org/10.1016/j.ultramic.2016.09.002
