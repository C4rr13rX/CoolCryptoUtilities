from __future__ import annotations
import json
import os
from django.http import JsonResponse
from django.views import View
from tools.wizard_learning.self_design import SCENE, assets, current, normalize, record
from tools.wizard_learning.store import LearningStore, brain_key

class WizardSelfDesignView(View):
    def get(self, request):
        if not getattr(request.user,"is_authenticated",False): return JsonResponse({"error":"Sign in required"},status=403)
        from .views import _brain_for_request
        profile=_brain_for_request(request); store=LearningStore(); key=brain_key(profile)
        return JsonResponse({"scene":SCENE,"design":current(store,key),"assets":assets(),"gate":store.gate(key,os.getenv("WIZARD_LEARNING_CHECKPOINT_ID","")),"learning":False})

    def post(self, request):
        if not getattr(request.user,"is_authenticated",False): return JsonResponse({"error":"Sign in required"},status=403)
        if len(request.body)>32768: return JsonResponse({"error":"Feedback exceeds 32 KB"},status=413)
        from .views import _brain_for_request
        profile=_brain_for_request(request); store=LearningStore(); key=brain_key(profile)
        try:
            body=json.loads(request.body); kind=body.get("kind")
            if kind not in ("visual-feedback","visual-observation"): raise ValueError("Unknown feedback kind")
            design=normalize(body.get("design",{}))
            payload={"design":design,"manifest":SCENE["foundation_manifest"],"observed":body.get("observed",{}),"feedback":str(body.get("feedback",""))[:2000],"ground_truth":False}
            if not isinstance(payload["observed"],dict): raise ValueError("Expected observation object")
            record(store,key,kind,payload)
            if body.get("learn") is True:
                # Reviewed human feedback can teach design preference; it is never
                # universal ethical compliance or a model self-certified reward.
                from tools.wizard_learning.brain import BrainClient, consolidate_reviewed
                review=body.get("reviewed_experience",{})
                result=consolidate_reviewed(BrainClient(profile),store,os.getenv("WIZARD_LEARNING_CHECKPOINT_ID",""),{"input":{"scene":SCENE,"experience":payload},"reviewed_experience":review},input_pool=int(os.getenv("WIZARD_FOUNDATION_INPUT_POOL","1")),outcome_pool=int(os.getenv("WIZARD_FOUNDATION_OUTCOME_POOL","4")))
                if result.get("consolidated") is not True:
                    return JsonResponse({"error":"Design outcome submitted without a positive acknowledgement. Learning may have occurred; reassess before retrying.","reassessment_required":True},status=503)
                return JsonResponse({"recorded":True,"consolidated":result.get("consolidated") is True,"reassessment_required":True})
            return JsonResponse({"recorded":True,"learning":False})
        except PermissionError as error: return JsonResponse({"error":str(error)},status=423)
        except (ValueError,TypeError,AttributeError) as error: return JsonResponse({"error":str(error)},status=400)
        except OSError as error: return JsonResponse({"error":str(error),"reassessment_required":True},status=502)
