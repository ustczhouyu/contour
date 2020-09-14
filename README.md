# Learning Panoptic Segmentation from Instance Contours 
## TODO:
* Functionalities
  - [x] Add output directory in configs. [Done]
  - [x] Add res5 supervision to hed. [Done]
  - [x] Add Contour Head branching from FPN Block. [Done]
  - [x] Add Dice to contour losses. [Done]
  - [x] Add Cityscapes panoptic evaluator. [Done]
  - [x] Refine instance using center regression. [Done]
  - [x] Add demo scripts that includes visualization for : [TBD]
    - [ ] semantic segmentation 
    - [ ] contour estimation
    - [ ] instance segmentation 
    - [x] panoptic segmentation
  - [ ] Add loss mask. Dilated with 5 pixels [Sep-15]

* Nice to have
  - [ ] Add NMS loss.
  - [ ] Add Contour performance evaluator.
  - [x] Add visualization for contours.

* Tests
  - [x] Verify contour predictions. [Done] [Binary-only]
  - [x] Verify Instance segmentation output. [Done]
  - [ ] Generate Videos. [Sep-20]

* Experiments 
  - [ ] Cityscapes
    - [ ] Loss_fn Ablation. (4 experiments)
        - [ ] Binary Contours R_50_1x 
            - [x] Weighted BCE [Done]
            - [x] Weighted BCE + Huber [Done]
            - [ ] Weighted BCE + Robust Loss [Sep-20]
            - [ ] Weighted BCE + NMS Loss [Sep-20]
    - [ ] GT Dilation Ablation with best loss_fn. (4 experiments)
        - [ ] Binay Contours R_50_1x
            - [ ] Dilation factor 1 [Sep-14]
            - [ ] Dilation factor 2 [Done]
            - [ ] Dilation factor 3 [Sepe-15]
    - [ ] Network Ablation with best dilation factor and best loss_fn. (3 experiments)
        - [ ] Binary Contours R_50_1x [Sep-10]
            - [ ] Seperate heads [Done]
            - [ ] Seperate Necks
        - [x] Multi-class Contours R_50_1x [Sep-10]
            - [x] Seperate heads [Done]
            - [ ] Seperate Necks
    - [ ] Postproc ignore pixel area Ablation. [Eval-only] [Sep-11]
        - [ ] Pixel Area 0 
        - [ ] Pixel Area 100
        - [ ] Pixel Area 300
        - [ ] Pixel Area 500
    - [ ] Best Network with R_101_3x against SOTA [Sep-11]
