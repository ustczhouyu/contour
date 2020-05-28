# Learning Panoptic Segmentation from Instance Contours 


* Seperate decoders:

               |-- fpn -- sem_seg_head --|
    Backbone --|                         |-- postproc -- instance -- panoptic
               |------- hed_head --------|

* Single decoder:

                      |-- sem_seg_head --|
    Backbone -- fpn --|                  |-- postproc -- instance -- panoptic
                      |-- contour_head --|

* Semantic with instance:

                                |-- seg --|
    Backbone - fpn -|- sem_seg -|         | - postproc - instance -- panoptic
                                |-contour-|

## TODO:
* Functionalities
 [x] Add output directory in configs. [May-27]
 [x] Add res5 supervision to hed. [May-27]
 [ ] Add Contour Head branching from FPN Block. [May-28]
 [ ] Add dice loss to contour losses. [May-29]
 [ ] Add Semantic with instance experiment. [May-30]
 [ ] Add Cityscapes panoptic evaluator. [May-31]
 [ ] Add configurable structuring element to generate gt contours. [Jun-01]
 [ ] Add support for COCO by generalizing postprocessing. [Jun-05]
 [ ] Add demo scripts that includes visualization for : [Jun-10]
     [ ] semantic segmentation 
     [ ] contour estimation
     [ ] instance segmentation 
     [ ] panoptic segmentation

* Tests
 [ ] Verify contour predictions. [May-27]
 [ ] Generate Videos. [Jun-15]

* Experiments 
 [ ] Cityscapes
    [ ] Preliminary Network Ablation with dilation factor 2.0 and Weighted BCE + Smooth L1. 
    Pick best network for next expeirments. (3 experiments)
        [x] HED Supervision Ablation
            [x] output [May-27]
            [x] output + res5 [May-27]
        [ ] Decoder Ablation with best supervision from above
            [ ] Binary Contours R_50_1x with Seperate Decoders
            [ ] Binary Contours R_50_1x with Single Decoder [May-29]
    [ ] Loss_fn Ablation. (7 experiments)
        [ ] Binary Contours R_50_1x 
            [ ] Weighted BCE [May-29]
            [ ] Weighted BCE + Smooth L1 [Done]
            [ ] Weighted BCE + Dice Loss [May-30]
            [ ] Weighted BCE + Smooth L1 + Dice Loss [May-30]
        [ ] Semantic with instance R_50_1x
            [ ] Cross Entropy [May-31]
            [ ] Focal Loss [May-31]
            [ ] Cross Entropy + Duality Loss [Jun-01]
            [ ] Focal Loss + Duality Loss [Jun-01]
    [ ] GT Dilation Ablation with best loss_fn. (4 experiments)
        [ ] Binary Contours R_50_1x
            [ ] Dilation factor 1 [Jun-02]
            [ ] Dilation factor 2 [Done]
            [ ] Dilation factor 3 [June-02]
        [ ] Semantic with instance R_50_1x
            [ ] Dilation factor 1 [Jun-03]
            [ ] Dilation factor 2 [Done]
            [ ] Dilation factor 3 [Jun-03]
    [ ] Network Ablation with best dilation factor and best loss_fn. (3 experiments)
        [ ] Binary Contours R_50_1x [Jun-04] [One of the below is Done.]
            [ ] Single Decoder 
            [ ] Seperate Decoders
        [ ] Multi-class Contours R_50_1x
            [ ] Single Decoder [Jun-05]
            [ ] Seperate Decoders [Jun-05]
        [ ] Semantic with instance contours class R_50_1x [Done]
    [ ] Postproc ignore pixel area Ablation. [Eval-only] [Jun-06]
        [ ] Pixel Area 0 
        [ ] Pixel Area 100
        [ ] Pixel Area 300
        [ ] Pixel Area 500
    [ ] Best Network with R_101_3x against SOTA [Jun-13]
[ ] COCO
    [ ] Network Ablation with best dilation factor and best loss_fn. (4 experiments)
        [ ] Binary Contours R_50_1x 
            [ ] Single Decoder [Jun-07]
            [ ] Seperate Decoders [Jun-08]
        [ ] Multi-class Contours R_50_1x
            [ ] Single Decoder [Jun-09]
            [ ] Seperate Decoders [Jun-10]
        [ ] Semantic with instance contours class R_50_1x [Jun-11]
    [ ] Postproc ignore pixel area Ablation. [Eval-only] [Jun-12]
        [ ] Pixel Area 0 
        [ ] Pixel Area 100
        [ ] Pixel Area 300
        [ ] Pixel Area 500
    [ ] Best Network with R_101_3x against SOTA [Jun-14] 
	

	


