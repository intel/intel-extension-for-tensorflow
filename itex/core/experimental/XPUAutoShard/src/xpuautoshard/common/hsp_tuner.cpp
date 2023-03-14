/* Copyright (c) 2023 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xpuautoshard/common/hsp_tuner.h"

namespace as {

HspAnnotationRef HspTuner::tune(GraphRef graph) {
  auto&& cost_model = getCostModel();
  auto&& best_score = cost_model->lowestScore();
  HspAnnotationRef best_annotation;
  do {
    (void)nextState();
    auto&& hsp_annotator = createAnnotator();
    auto&& annot = hsp_annotator->annotate(graph);
    auto&& score = cost_model->evaluate(graph, annot);
    if (*best_score < *score) {
      best_score = score;
      best_annotation = annot;
    }
    updateScore(score);
  } while (!stopCriterionMet());
  return best_annotation;
}

}  // namespace as
