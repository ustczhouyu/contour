# Learning Panoptic Segmentation from Instance Contours 
## TODO:
* Functionalities
  - [x] Add output directory in configs. [May-27]
  - [x] Add res5 supervision to hed. [May-27]
  - [x] Add Contour Head branching from FPN Block. [May-28]
  - [x] Add Dice to contour losses. [Jun-06]
  - [ ] Add loss mask. Dilated with 5 pixels [Jun-08]
  - [ ] Merge close contours [Jun-08]
  - [ ] Add Cityscapes panoptic evaluator. [Jun-08]
  - [ ] Add support for COCO by generalizing postprocessing. [Jun-10]
  - [ ] Add demo scripts that includes visualization for : [Jun-10]
    - [ ] semantic segmentation 
    - [ ] contour estimation
    - [ ] instance segmentation 
    - [ ] panoptic segmentation

* Nice to have
  - [ ] Add NMS loss.
  - [ ] Add Contour performance evaluator.
  - [x] Add visualization for contours.

* Tests
  - [x] Verify contour predictions. [May-27] [Binary-only]
  - [x] Verify Instance segmentation output. [May-28]
  - [ ] Generate Videos. [Jun-15]

* Experiments 
  - [ ] Cityscapes
    - [ ] Loss_fn Ablation. (4 experiments)
        - [ ] Binary Contours R_50_1x 
            - [x] Weighted BCE [Jun-07]
            - [x] Weighted BCE + Huber [Done]
            - [ ] Weighted BCE + Dice [Jun-08]
            - [ ] Weighted BCE + Huber + Dice [Jun-08]
    - [ ] GT Merged Contour Ablation with best loss_fn. (4 experiments)
        - [ ] Binay Contours R_50_1x
            - [ ] Merged contours [Jun-09]
            - [ ] Unmerged contours [Jun-09]
    - [ ] GT Dilation Ablation with best loss_fn. (4 experiments)
        - [ ] Binay Contours R_50_1x
            - [ ] Dilation factor 1 [Jun-10]
            - [ ] Dilation factor 2 [Done]
            - [ ] Dilation factor 3 [June-10]
    - [ ] Network Ablation with best dilation factor and best loss_fn. (3 experiments)
        - [ ] Binary Contours R_50_1x [Jun-10] [
            - [ ] Seperate heads [Done]
            - [ ] Seperate Necks
        - [x] Multi-class Contours R_50_1x [Jun-10]
            - [x] Seperate heads [Done]
            - [ ] Seperate Necks
    - [ ] Postproc ignore pixel area Ablation. [Eval-only] [Jun-11]
        - [ ] Pixel Area 0 
        - [ ] Pixel Area 100
        - [ ] Pixel Area 300
        - [ ] Pixel Area 500
    - [ ] Best Network with R_101_3x against SOTA [Jun-11]
  - [ ] COCO
    - [ ] Network Ablation with best dilation factor and best loss_fn. (4 experiments)
        - [ ] Binary Contours R_50_1x [Jun-12]
            - [ ] Seperate heads [Done]
            - [ ] Seperate Necks
        - [ ] Multi-class Contours R_50_1x [Jun-14]
            - [ ] Seperate heads [Done]
            - [ ] Seperate Necks
    - [ ] Postproc ignore pixel area Ablation. [Eval-only] [Jun-14]
        - [ ] Pixel Area 0 
        - [ ] Pixel Area 100
        - [ ] Pixel Area 300
        - [ ] Pixel Area 500
    - [ ] Best Network with R_101_3x against SOTA [Jun-16] 
