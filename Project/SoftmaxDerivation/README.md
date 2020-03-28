# Softmax Derivation

## Derivation

* [**Softmax Classification Back Propagation with Cross Entropy**](Softmax_Derivation.md)

## Implementation

### Single Layer & Binary Classification

```sh
python3 Softmax_Derivation.py
```

### Multi-Layer & Multi-class Classification

```sh
python3 Softmax_Derivation_Multiple.py
```

## Notes

* [Softmax - Activation Function](../../Notes/Element/Activation_Function.md#Softmax)
* [Cross Entropy - Loss Function](../../Notes/Element/Loss_Function.md#Cross-Entropy)

## Resources

* [Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
  * calculate cross-entropy with for loop way (two sigma formula) - for each batch & each category
* [Classification and Loss Evaluation - Softmax and Cross Entropy Loss](https://deepnotes.io/softmax-crossentropy)
  * explain the stable version of softmax
  * calculate cross-entropy with matrix way (don't need one-hot label)
