# Dialogue

## Overview

### Catetory

* **Task-oriented** dialogue: to get something done during conversation
  * Assistive
    * customer service
    * giving recommendations
    * question answering
  * Co-operative
    * two agents solve a task together through dialogue
  * Adversarial
    * two agents compete in a task through dialogue
* **Social** dialogue: no explicit task
  * Chit-chat
    * for fun or company
  * Therapy / mental wellbeing

### Approach

* pre-neural dialogue system
  * pre-defined templates
  * retrieve an appropriate response from a corpus of responses
* open-ended freeform dialogue system

## Problems / Solution

A naive application of standard seq2seq+attention methods thas serious pervasive deficiencies for (chitchat) dialogue

* Genericness / boring responses
* Irrelevant responses (not sufficiently related to context)
* Repetition
* Lack of context (not remembering conversation history)
* Lack of consistent persona

### Irrelevant response problem

* [[1510.03055] A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/abs/1510.03055)

### Genericness / boring response problem

... cs224n lecture 15 slides

## Resources

* [[1506.05869] A Neural Conversational Model](https://arxiv.org/abs/1506.05869)
* [Neural Responding Machine for Short-Text Conversation - ACL Anthology](https://www.aclweb.org/anthology/P15-1152/)
