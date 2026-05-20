### Usage and Description

This dataset is intended to be used to train or test computer vision models in detecting defects of welding seams using conventional camera images (Not HDR, Radiography or X-Ray images), it is meant as an quick and cheap alternative of visual inspection using readily available tools (like a smartphone).

This dataset consists of frames extracted from video recordings and photographs of welding seams. Each image is annotated using the YOLO annotation format. The dataset includes the following classes of welding seam defects, ordered by class IDs from 0 to 5.

This dataset is roughly distributed to 85% train, 10% valid, and 5% test. You may inspect the images with bounding boxes on the 'preview' directory.

#### Defect Classes:

1. **Class 0: Adjacent Defects (`adj`)**
   - **Spatter**: Small droplets of molten metal expelled from the weld pool, which solidify on the surface of the workpiece.
   - **Arc Strikes**: Localized melting or surface damage caused by the welding arc striking the workpiece outside the intended weld area.

2. **Class 1: Integrity Defects (`int`)**
   - **Crater Cracks**: Depressions at the end of a weld bead that can be points of weakness.
   - **Slag Inclusions**: Non-metallic by-products from welding trapped in the weld metal.
   - **Blisters**: Raised areas on the weld surface caused by trapped gas.
   - **Porosity**: Small holes in the weld metal formed by trapped gas during solidification.
   - **Burn-Through**: Excessive penetration of the weld metal, resulting in holes in the base metal.
   - **Inclusions**: Foreign materials, such as slag or tungsten, trapped within the weld metal.

3. **Class 2: Geometry Defects (`geo`)**
   - **Undercut**: Grooves melted into the base metal adjacent to the weld toe or root that are left unfilled by weld metal.
   - **Lack of Fusion**: Failure of the weld metal to properly fuse with the base metal or preceding weld bead.
   - **Overlap**: Weld metal that flows over the base metal surface without bonding.
   - **Surface Irregularities**: Roughness or unevenness on the weld surface caused by improper welding technique.
   - **Concavity/Sinking**: Depressions or concave areas in the weld surface.
   - **Uneven Bead Shape**: Irregular weld bead shape or height.

4. **Class 3: Post-Processing Defects (`pro`)**
   - **Burrs**: Sharp edges or projections remaining on the weld after processing.
   - **Improper End Finishing**: Rough or uneven end face of the weld.
   - **Scratches**: Surface damage caused by mechanical means after welding.
   - **Dents**: Depressions or indentations in the weld surface from impacts or pressure.

5. **Class 4: Non-Fulfillment Defects (`non`)**
   - **Incomplete Penetration**: Weld metal that fails to extend through the thickness of the joint.
   - **Lack of Fusion**: Larger areas where the weld metal fails to fuse properly.

#### Dataset Source

This dataset was a modified version of this dataset:
https://www.kaggle.com/datasets/alexlaym/weld-defects
uploaded by Eduard Ghanza

Modifications are sourced from this dataset:
https://universe.roboflow.com/welding-2bplp/weld-quality-inspection-rei9l/dataset/9


### FREE PALESTINE 🔻