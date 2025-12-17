# Classifier Migration TODOs

This list tracks the migration of various classifiers to inherit from `RuleBasedClassifier`.

## Classifiers to Migrate to RuleBasedClassifier

*   **ProgressBarClassifier**: Currently `LabelClassifier`. Requires access to `ClassificationResult` for finding overlapping blocks and page number proximity.
*   **PreviewClassifier**: Currently `LabelClassifier`. Requires custom logic for grouping drawings and checking internal elements.

## Classifiers to Check for Migration Suitability

*   **LoosePartSymbolClassifier**: Still needs to be evaluated.
*   **DiagramClassifier**: Still needs to be evaluated.
*   **ArrowClassifier**: Still needs to be evaluated.
*   **SubAssemblyClassifier**: Still needs to be evaluated.

## Complex/Composite Classifiers (Likely won't migrate to RuleBasedClassifier directly)

*   **PartsClassifier**: Combines candidates from other classifiers.
*   **PartsListClassifier**: Operates on groups of blocks.
*   **StepClassifier**: Combines candidates from multiple other classifiers.
*   **TriviaTextClassifier**: Operates on clusters of text blocks.
*   **ScaleClassifier**: Involves searching for containers and internal candidates.

## Already RuleBasedClassifier (No action needed)

*   BackgroundClassifier
*   ProgressBarIndicatorClassifier
*   DividerClassifier
*   BagNumberClassifier
*   PartCountClassifier
*   PartNumberClassifier
*   StepNumberClassifier
*   StepCountClassifier
*   PieceLengthClassifier
*   PartsImageClassifier
*   ShineClassifier
