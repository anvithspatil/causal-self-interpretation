# Causal Self-Interpretation in Language Models

A mechanistic interpretability study investigating whether a language model's internal representations correspond to the variables it reports as determining its outputs.

## Research Question

When a language model solves a deterministic task and reports which variable determines its answer, does its internal representation encode that same factor?

## Core Idea

The experiment compares model self-report, internal representations, and causal activation interventions.

## Model

**Qwen/Qwen3-4B**

The experiments access hidden states across 36 transformer layers and search the final 12 layers.

## V5 Results

- V5.2 best layer: **34**
- Self-report accuracy: **100%** on 6 held-out examples
- Internal prediction accuracy: **100%**
- Self-report/internal alignment: **100%**

## Experimental Pipeline

Deterministic synthetic tasks → factor manipulation → hidden-state extraction → representation localization → causal intervention → behavioral measurement → self-report comparison.

## Project Structure

data/ — synthetic task data
v5/ — model loading and experiments
results/ — recorded experimental results
paper/ — research write-up

## Reproducibility

The repository contains the experimental scripts and recorded JSON results. Model weights are not included.

## Limitations

The evaluation uses a small synthetic dataset and one model. These results should be treated as a proof-of-concept rather than a general claim about language models.

## Related Work

A follow-up project, **Alignment Surviving Model Modification**, tests whether the observed representation–self-report alignment persists after a controlled model modification.

## Status

Research prototype / experimental implementation.
