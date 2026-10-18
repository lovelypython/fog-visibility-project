# The Foggy Cityscapes Dataset

*Foggy Cityscapes* derives from the [Cityscapes][cityscapes] dataset and constitutes a collection of partially synthetic foggy images that automatically inherit the semantic annotations of their real, clear-weather counterparts in Cityscapes.

These foggy images are generated from the clear-weather images of Cityscapes with the fog simulation pipeline that we present in our work ["Semantic Foggy Scene Understanding with Synthetic Data"][project_page]. This pipeline involves computation of a denoised and complete depth map as well as a corresponding transmittance map, the latter being combined with the real clear-weather image to obtain the synthetic foggy image.

Due to licensing issues, the main dataset modality of foggy images is only available for download at the website of the original Cityscapes dataset:
www.cityscapes-dataset.net
This is also the case for pixel-level semantic annotations of the images as well as other modalities which are shared by *Foggy Cityscapes* and Cityscapes (e.g. camera parameters). On the contrary, the auxiliary *Foggy Cityscapes* modalities of **denoised depth maps and transmittance maps are available at our website**:
people.ee.ethz.ch/~csakarid/SFSU_synthetic

In the following, we provide details both on the foggy images and the depth and transmittance maps of *Foggy Cityscapes*.


### Dataset Structure

The folder structure of *Foggy Cityscapes* follows that of Cityscapes, with minor extensions and modifications. It is summarized as follows:
```
{root}/{type}/{split}/{city}/{city}_{seq:0>6}_{frame:0>6}_{type}{ext}
```
Please refer to the README file of the Cityscapes git repository for a detailed presentation of this structure: 
https://github.com/mcordts/cityscapesScripts
In the following, we outline only those elements of the folder structure that are potentially differentiated in *Foggy Cityscapes*.

The meaning of the individual elements is:
 - `root`  the root folder of the Foggy Cityscapes dataset. We recommend that this folder coincides with the root folder of the original Cityscapes dataset.
 - `type`  the type/modality of data, e.g. `depth_stereoscopic` for completed depth maps, or `leftImg8bit` for left 8-bit foggy images.
 - `split` the split, i.e. `train`/`val`/`test`/`train_extra`.
 - `ext`   the extension of the file and optionally a suffix, e.g. `.mat` for completed depth maps.

Possible values of `type`
 - `leftImg8bit`               the left foggy images in 8-bit format. These are the foggy versions of the standard annotated Cityscapes images.
 - `depth_stereoscopic`        denoised and complete depth maps, based on the precomputed disparity maps (`disparity` type) included in Cityscapes. These depth maps are stored in MATLAB `.mat` files in `double` format and **directly encode depth values in meters**.
 - `leftImg8bit_transmittance` the transmittance maps for the left foggy images, in 8-bit format. Value `0` corresponds to transmittance `0`, while value `255` corresponds to transmittance `1`.

### Multiple versions with different fog densities

We provide three different versions of *Foggy Cityscapes* for the `train`/`val`/`test` splits, each corresponding to a distinct fog density. The fog density is regulated by the **attenuation coefficient** `beta` (measured in inverse meters) which is involved in the standard optical model for fog; each version is characterized by a common, constant value of `beta` across all images.

In order to distinguish between images of different versions, the extension field `ext` of the types `leftImg8bit` and `leftImg8bit_transmittance` is modified to declare the value of `beta` that was used to generate each image, as shown in the following sample image names:
```
erfurt_000000_000019_leftImg8bit_foggy_beta_0.01.png
erfurt_000000_000019_leftImg8bit_transmittance_beta_0.01.png
```
Depth values are independent of fog density, so no such modification is applied to the names of depth map files.

Foggy images in the `train`/`val`/`test` splits are available for `beta` equal to `0.005`, `0.01`, and `0.02`, while those in the `train_extra` split are available for `beta` equal to `0.01`.


### Foggy Cityscapes-refined

A refined list of 550 Cityscapes images (498 `train` plus 52 `val`) that yield high-quality synthetic foggy images is provided in the file `foggy_trainval_refined_filenames.txt`. Details on the refinement criteria are given in our [publication][project_page].


### Citation

If you use *Foggy Cityscapes* in your work, please cite both our publication as listed on our [website][project_page] and the [Cityscapes publication][cityscapes_citation].

### Trained beta recognition program

The project includes a trained CNN that classifies a **transmittance map** into one of the dataset's three fog attenuation coefficients (`beta = 0.005`, `0.01`, or `0.02 m^-1`). It accepts grayscale `leftImg8bit_transmittance` PNGs, not ordinary RGB fog photographs.

Recognize one map:
```bash
python3 recognize_transmittance.py leftImg8bit_transmittance/val/lindau/lindau_000029_000019_leftImg8bit_transmittance_beta_0.005.png
```

Recognize every PNG below a directory and save results:
```bash
python3 recognize_transmittance.py leftImg8bit_transmittance/val --csv outputs/val_predictions.csv --json outputs/val_predictions.json
```

The program loads `outputs/transmittance_beta/best_model.pt` by default. Use `--model PATH` to select another checkpoint. Install `torch`, `numpy`, and `Pillow` if they are not already available. To retrain the model, run:
```bash
python3 train_transmittance_beta.py
```

The physical model is Beer–Lambert: `t = exp(-beta*d)`, so `-ln(t) = beta*d`. Since this auxiliary dataset contains transmittance maps but no depth maps, the trained program predicts the beta category; it cannot independently infer metric depth from transmittance alone. The saved evaluation metrics report 93.0% validation accuracy and 92.46% test accuracy for this training run.


### License

*Foggy Cityscapes* is made available for non-commercial use under the license agreement which is contained in the attached file LICENSE.txt


### Contact

Christos Sakaridis
csakarid@vision.ee.ethz.ch
people.ee.ethz.ch/~csakarid/SFSU_synthetic

[project_page]: <http://people.ee.ethz.ch/~csakarid/SFSU_synthetic/>
[cityscapes]: <https://www.cityscapes-dataset.com/>
[cityscapes_citation]: <https://www.cityscapes-dataset.com/citation/>
