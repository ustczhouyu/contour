"""Cityscapes Evaluator Script.

Adapted from: https://github.com/facebookresearch/detectron2/
              blob/master/detectron2/evaluation/cityscapes_evaluation.py
"""
import contextlib
import glob
import io
import itertools
import json
import logging
import os
import tempfile
from collections import OrderedDict

import numpy as np
import torch
from cityscapesscripts.evaluation.evalPanopticSemanticLabeling import \
    evaluatePanoptic
from cityscapesscripts.helpers.labels import name2label, trainId2label
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils import comm
from fvcore.common.file_io import PathManager
from panopticapi.utils import id2rgb
from PIL import Image
from tabulate import tabulate


class CityscapesEvaluator(DatasetEvaluator):
    """Base class for evaluation using cityscapes API."""

    def __init__(self, dataset_name):
        """Intialize the evaluator.

        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        # pylint: disable=no-member
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._working_dir = None
        self._temp_dir = None
        self._predictions = []

    def reset(self):
        """Reset evaluator."""
        self._predictions = []
        self._working_dir = tempfile.TemporaryDirectory(
            prefix="cityscapes_eval_")
        self._temp_dir = self._working_dir.name
        # All workers will write to the same results directory
        # pylint: disable=fixme
        # TODO this does not work in distributed training
        self._temp_dir = comm.all_gather(self._temp_dir)[0]
        if self._temp_dir != self._working_dir.name:
            self._working_dir.cleanup()
        # pylint: disable=logging-format-interpolation
        self._logger.info(
            "Writing cityscapes results to temporary directory {} ...".format(
                self._temp_dir)
        )


# pylint: disable=too-many-locals
class CityscapesInstanceEvaluator(CityscapesEvaluator):
    """Evaluate instance segmentation results using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def process(self, inputs, outputs):
        """Prepare the inputs and outputs for metric evaluation."""
        for _input, _output in zip(inputs, outputs):
            file_name = _input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_txt = os.path.join(self._temp_dir, basename + "_pred.txt")

            output = _output["instances"].to(self._cpu_device)
            num_instances = len(output)
            with open(pred_txt, "w") as fout:
                for i in range(num_instances):
                    pred_class = output.pred_classes[i]
                    classes = self._metadata.thing_classes[pred_class]
                    class_id = name2label[classes].id
                    score = output.scores[i]
                    mask = output.pred_masks[i].numpy().astype("uint8")
                    png_filename = os.path.join(
                        self._temp_dir, basename +
                        "_{}_{}.png".format(i, classes)
                    )

                    Image.fromarray(mask * 255).save(png_filename)
                    fout.write("{} {} {}\n".format(
                        os.path.basename(png_filename), class_id, score))

    # pylint: disable=inconsistent-return-statements
    def evaluate(self):
        """Evaluate the predictions and compute results.

        Returns:
            dict: has a key "segm", whose value is a dict of "AP" and "AP50".
        """
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # pylint: disable=import-outside-toplevel
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval

        # pylint: disable=logging-format-interpolation
        self._logger.info(
            "Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.distanceAvailable = True
        cityscapes_eval.args.gtInstancesFile = os.path.join(
            self._temp_dir, "gtInstances.json")

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/
        # cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        ground_truth_img_list = glob.glob(os.path.join(
            gt_dir, "*", "*_gtFine_instanceIds.png"))
        # pylint: disable=len-as-condition
        assert len(
            ground_truth_img_list
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        prediction_img_list = []
        for ground_truth in ground_truth_img_list:
            prediction_img_list.append(cityscapes_eval.getPrediction(ground_truth,
                                                                     cityscapes_eval.args))
        results = cityscapes_eval.evaluateImgLists(prediction_img_list,
                                                   ground_truth_img_list,
                                                   cityscapes_eval.args)["averages"]

        ret = OrderedDict()
        ret["segm"] = {"AP": results["allAp"] * 100,
                       "AP50%": results["allAp50%"] * 100,
                       "AP50m": results["allAp50m"] * 100,
                       "AP50%50m": results["allAp100m"] * 100,
                       "AP100m": results["allAp"] * 100}
        self._working_dir.cleanup()
        return ret


class CityscapesSemSegEvaluator(CityscapesEvaluator):
    """Evaluate semantic segmentation results using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def process(self, inputs, outputs):
        """Prepare the inputs and outputs for metric evaluation."""
        for _input, _output in zip(inputs, outputs):
            file_name = _input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_filename = os.path.join(
                self._temp_dir, basename + "_pred.png")

            output = _output["sem_seg"]
            pred = 255 * np.ones(output.shape, dtype=np.uint8)
            for train_id, label in trainId2label.items():
                if label.ignoreInEval:
                    continue
                pred[output == train_id] = label.id
            Image.fromarray(pred).save(pred_filename)

    # pylint: disable=inconsistent-return-statements
    def evaluate(self):
        """Evaluate the predictions and compute results."""
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        # pylint: disable=import-outside-toplevel
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

        # pylint: disable=logging-format-interpolation
        self._logger.info(
            "Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/
        # cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        ground_truth_img_list = glob.glob(
            os.path.join(gt_dir, "*", "*_gtFine_labelIds.png"))
        # pylint: disable=len-as-condition
        assert len(
            ground_truth_img_list
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        prediction_img_list = []
        for ground_truth in ground_truth_img_list:
            prediction_img_list.append(cityscapes_eval.getPrediction(cityscapes_eval.args,
                                                                     ground_truth))
        results = cityscapes_eval.evaluateImgLists(prediction_img_list,
                                                   ground_truth_img_list,
                                                   cityscapes_eval.args)
        ret = OrderedDict()
        ret["sem_seg"] = {
            "IoU": 100.0 * results["averageScoreClasses"],
            "iIoU": 100.0 * results["averageScoreInstClasses"],
            "IoU_sup": 100.0 * results["averageScoreCategories"],
            "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
        }
        self._working_dir.cleanup()
        return ret


class CityscapesPanopticEvaluator(CityscapesEvaluator):
    """Evaluate Panoptic Quality metrics on Cityscapes using cityscapesscripts."""

    def _convert_category_id(self, segment_info):
        """Convert stuff/thing id to cityscapes category id."""
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            classes = self._metadata.thing_classes[segment_info["category_id"]]
        else:
            classes = self._metadata.stuff_classes[segment_info["category_id"]]
        segment_info["category_id"] = name2label[classes].id
        return segment_info

    def process(self, inputs, outputs):
        """Prepare the inputs and outputs for metric evaluation."""
        for _input, _output in zip(inputs, outputs):
            panoptic_img, segments_info = _output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()

            file_name = _input["file_name"]
            file_name = os.path.splitext(os.path.basename(file_name))[0]
            image_id = '_'.join(file_name.split('_')[:-1])
            pred_filename = image_id + "_pred.png"

            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(
                    x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": image_id,
                        "file_name": pred_filename,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    # pylint: disable=inconsistent-return-statements
    def evaluate(self):
        """Evaluate the predictions and compute results."""
        comm.synchronize()
        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            # pylint: disable=logging-format-interpolation
            self._logger.info(
                "Writing all panoptic predictions to {} ...".format(pred_dir))
            for prediction in self._predictions:
                with open(os.path.join(pred_dir, prediction["file_name"]), "wb") as file:
                    file.write(prediction.pop("png_string"))

            with open(gt_json, "r") as file:
                json_data = json.load(file)
            json_data["annotations"] = self._predictions
            predictions_json = os.path.join(
                pred_dir, "cityscapes_panoptic_val.json")
            results_json = os.path.join(
                pred_dir, "resultPanopticSemanticLabeling.json")
            with PathManager.open(predictions_json, "w") as file:
                file.write(json.dumps(json_data))

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = evaluatePanoptic(
                    gt_json,
                    pred_json_file=PathManager.get_local_path(
                        predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                    resultsFile=PathManager.get_local_path(results_json)
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        self._print_panoptic_results(pq_res)

        return results

    def _print_panoptic_results(self, pq_res):
        """Print panoptic results."""
        headers = ["", "PQ", "SQ", "RQ", "#categories"]
        data = []
        for name in ["All", "Things", "Stuff"]:
            row = [name] + [pq_res[name][k] *
                            100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe",
            floatfmt=".3f", stralign="center", numalign="center"
        )
        # pylint: disable=logging-not-lazy
        self._logger.info("Panoptic Evaluation Results:\n" + table)
