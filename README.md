# Learning Panoptic Segmentation from Instance Contours 
## TODO:
* Functionalities
  - [x] Add output directory in configs. [May-27]
  - [x] Add res5 supervision to hed. [May-27]
  - [ ] Add Contour Head branching from FPN Block. [May-28]
  - [ ] Add Dice to contour losses. [May-29]
  - [ ] Add Semantic with instance experiment. [May-30]
  - [ ] Add Cityscapes panoptic evaluator. [May-31]
  - [ ] Add configurable structuring element to generate gt contours. [Jun-01]
  - [ ] Add support for COCO by generalizing postprocessing. [Jun-05]
  - [ ] Add demo scripts that includes visualization for : [Jun-10]
    - [ ] semantic segmentation 
    - [ ] contour estimation
    - [ ] instance segmentation 
    - [ ] panoptic segmentation

* Nice to have
  - [ ] Add NMS loss.
  - [ ] Add Contour performance evaluator.
  - [ ] Add visualization in HED/SBD format for contours.

* Tests
  - [x] Verify contour predictions. [May-27] [Binary-only]
  - [ ] Verify Instance segmentation output. [May-28]
  - [ ] Generate Videos. [Jun-15]

* Experiments 
  - [ ] Cityscapes
    - [ ] Preliminary Network Ablation with dilation factor 2.0 and Weighted BCE + Huber. 
        Pick best network for next expeirments. (4 experiments)
        - [x] HED Supervision Ablation
            - [x] output [May-27]
            - [x] output + res5 [May-27]
        - [ ] Decoder Ablation with best supervision from above
            - [ ] Binary Contours R_50_1x with Seperate Decoders (HED + FPN)
            - [ ] Binary Contours R_50_1x with Seperate Decoders (2 FPN Heads) [May-28]
            - [ ] Binary Contours R_50_1x with Single Decoder [May-28]
    - [ ] Loss_fn Ablation. (7 experiments)
        - [ ] Binary Contours R_50_1x 
            - [ ] Weighted BCE [May-29]
            - [ ] Weighted BCE + Huber [Done]
            - [ ] Weighted BCE + Dice [May-29]
            - [ ] Weighted BCE + Huber + Dice [May-29]
        - [ ] Semantic with instance R_50_1x
            - [ ] Cross Entropy [May-30]
            - [ ] Focal Loss [May-30]
            - [ ] Cross Entropy + Duality Loss [May-31]
            - [ ] Focal Loss + Duality Loss [May-31]
    - [ ] GT Dilation Ablation with best loss_fn. (4 experiments)
        - [ ] Binary Contours R_50_1x
            - [ ] Dilation factor 1 [Jun-02]
            - [ ] Dilation factor 2 [Done]
            - [ ] Dilation factor 3 [June-02]
        - [ ] Semantic with instance R_50_1x
            - [ ] Dilation factor 1 [Jun-03]
            - [ ] Dilation factor 2 [Done]
            - [ ] Dilation factor 3 [Jun-03]
    - [ ] Network Ablation with best dilation factor and best loss_fn. (3 experiments)
        - [ ] Binary Contours R_50_1x [Jun-04] [One of the below is Done.]
            - [ ] Single Decoder 
            - [ ] Seperate Decoders
        - [ ] Multi-class Contours R_50_1x
            - [ ] Single Decoder [Jun-05]
            - [ ] Seperate Decoders [Jun-05]
        - [ ] Semantic with instance contours class R_50_1x [Done]
    - [ ] Postproc ignore pixel area Ablation. [Eval-only] [Jun-06]
        - [ ] Pixel Area 0 
        - [ ] Pixel Area 100
        - [ ] Pixel Area 300
        - [ ] Pixel Area 500
    - [ ] Best Network with R_101_3x against SOTA [Jun-13]
  - [ ] COCO
    - [ ] Network Ablation with best dilation factor and best loss_fn. (4 experiments)
        - [ ] Binary Contours R_50_1x 
            - [ ] Single Decoder [Jun-07]
            - [ ] Seperate Decoders [Jun-08]
        - [ ] Multi-class Contours R_50_1x
            - [ ] Single Decoder [Jun-09]
            - [ ] Seperate Decoders [Jun-10]
        - [ ] Semantic with instance contours class R_50_1x [Jun-11]
    - [ ] Postproc ignore pixel area Ablation. [Eval-only] [Jun-12]
        - [ ] Pixel Area 0 
        - [ ] Pixel Area 100
        - [ ] Pixel Area 300
        - [ ] Pixel Area 500
    - [ ] Best Network with R_101_3x against SOTA [Jun-14] 
