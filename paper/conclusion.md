
This study investigated whether a language model's verbal causal self-attributions correspond to information encoded in its internal representations.

Using Qwen3-4B and controlled deterministic systems, we separated behavioral self-report, representation analysis, and causal intervention into distinct experimental stages.

The final V5.2 experiment found 100% agreement between the model's verbal causal attribution and the factor predicted from its internal activation representation across six held-out examples.

However, activation-patching experiments did not establish that the identified representations were causally sufficient for the model's behavior.

The central finding is therefore representational rather than causal:
this controlled setting, the model's verbal causal attributions aligned with factor-discriminative information in its internal representations.

This distinction is important for mechanistic interpretability.
Information about a concept being present in an activation does not automatically mean that the activation is the mechanism responsible for the model's behavior.

Future work should test whether this alignment survives larger evaluation sets, different models, natural-language reasoning tasks, and stronger causal interventions.

Overall, the project provides a reproducible framework for studying the relationship between what language models say about their own reasoning and what can be recovered from their internal representations.