## Hangul Reader

Aim of the project is to build a CNN that can detect the jamo (letters) in a handwritten Korean word,
and inform the user how to pronounce the word.

Will be loosely based on this fantastic project:

https://github.com/IBM/tensorflow-hangul-recognition

However their aim was simply to recognise a character, so they trained purely on the characters.
The difference with this project is detecting the underlying letters, and thus being able to
find the pronounciation of a word/character the model has never seen before, based purely
on the positioning of the jamo.