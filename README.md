## Description

* Goal
    - Analyze the top 1,000 questions from r/askreddit and find similar threads

* Sample use cases:
    - Recommend similar questions when visiting a thread
    - Identify reposts

* Method
    - Convert text to vector using bag-of-word method. The tally can be either
        - (1) raw word frequency, or
        - (2) tf-idf
    - Use cosine similarity metric to find the closest matching thread name

* Observations (wip)
    - Thread names are usually short. Therefore using raw word count will yield higher similarity score than tf-idf because of repeated phrases (eg. what, you(r), etc.)
    - tf-idf works slightly better because rarer topics can be matched


## Results
#### Compare similarity score

![](./data/compare%20sim%20score.png)

#### Preview data


**Using raw frequency count**
| thread1                                                                                  | thread2                                                                                      |   similarity |
|:-----------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|-------------:|
| what would you do if you woke up in the 80's?                                            | which celebr went on the fastest, hardest and most destruct downward spiral from their peak? |     0        |
| what is your strang turn-on?                                                             | what suck about turn 30?                                                                     |     1        |
| what wors than a wet handshake?                                                          | what is 100% wors when wet?                                                                  |     0.707107 |
| zoo worker of reddit, what is the dumbest thing someon has ask about an animal?          | what would you do if someon came up and ask if they could give you a hug?                    |     0.632456 |
| what is your favorit bromanc in fiction?                                                 | who are your favorit fiction couple?                                                         |     1        |
| women of reddit, what the stupidest excus a man has ever given you to not wear a condom? | women of reddit, what is the grossest thing a man has said to you?                           |     0.774597 |
| the world is now under the iron fist of canada. what changes?                            | what a piec of inform that the world is not readi for?                                       |     0.707107 |
| what a song that everybodi knows?                                                        | what song lyric is forev stuck in your head?                                                 |     1        |
| when it come to date apps, what is an automat “pass” for you?                            | what must one never do on a first date?                                                      |     0.57735  |
| who someon you look up to or idol as a kid that you now can't stand?                     | if you had to name your kid after a disease/med condit what would you name them?             |     0.57735  |


**Using raw tf-idf**
| thread1                                                                                  | thread2                                                            |   similarity |
|:-----------------------------------------------------------------------------------------|:-------------------------------------------------------------------|-------------:|
| what would you do if you woke up in the 80's?                                            | what your favorit 80 movie?                                        |     0.436622 |
| what is your strang turn-on?                                                             | what the weirdest thing that turn you on?                          |     0.412156 |
| what wors than a wet handshake?                                                          | what is 100% wors when wet?                                        |     0.665395 |
| zoo worker of reddit, what is the dumbest thing someon has ask about an animal?          | what the dumbest thing you believ as a child?                      |     0.266961 |
| what is your favorit bromanc in fiction?                                                 | who is your favorit fiction doctor?                                |     0.536878 |
| women of reddit, what the stupidest excus a man has ever given you to not wear a condom? | women of reddit, what is the grossest thing a man has said to you? |     0.333547 |
| the world is now under the iron fist of canada. what changes?                            | what is the most disturb fact you know about canada?               |     0.275582 |
| what a song that everybodi knows?                                                        | what do you hate that everybodi seem to love?                      |     0.394317 |
| when it come to date apps, what is an automat “pass” for you?                            | what must one never do on a first date?                            |     0.422814 |
| who someon you look up to or idol as a kid that you now can't stand?                     | who is a famous singer that you cannot stand their sing voice?     |     0.245291 |