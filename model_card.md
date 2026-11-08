# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

This card compares both models.

## 1. Model Overview

**Model type:**
I compared both models: the rule based model and the ML model.

**Intended purpose:**
The Mood Machine labels short, social-media-style text posts with one of four moods: `positive`, `negative`, `neutral`, or `mixed`. It is a **learning project**. It is meant to show how a simple text classifier works and where it breaks.

It should **not** be used for:
- Judging how a real person is feeling, or checking anyone's mental health
- Moderating content, screening job candidates or students, or making any decision that affects someone
- Any setting where a wrong label could cause harm, such as missing a message from someone in distress

**How it works (brief):**
- **Rule based:** The text is lowercased and split into words. Each word on a positive word list adds +1 and each word on a negative word list subtracts 1. A negation word ("not", "never", ...) flips the next word. A post with both positive and negative words is labeled `mixed`. Otherwise the sign of the score decides the label.
- **ML:** The text is turned into word counts (bag of words). A logistic regression model learns from the labeled examples which words tend to go with which label.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` in `dataset.py` contains **13 posts**. The starter code had 6. I added **7 new posts** in Part 1 to cover more language styles:

| Post | Label | Style |
|---|---|---|
| Lowkey stressed but kind of proud of myself | mixed | slang, mixed emotions |
| That exam was no cap the best grade I've ever gotten 😂 | positive | slang, emoji |
| I absolutely love getting stuck in traffic for two hours 🙃 | negative | sarcasm, emoji |
| Just got home, making dinner now | neutral | plain update |
| Missing my friends but this new city is growing on me 🥲 | mixed | mixed emotions, emoji |
| ugh my phone died again :( | negative | informal, text emoticon |
| It's whatever, I guess | neutral | subtle tone |

**Labeling process:**
I assigned every label **by hand**, using my own reading of each post. Each post in `SAMPLE_POSTS` has exactly one label at the same position in `TRUE_LABELS`, and both lists have 13 items.

Hard-to-label posts:
- **"It's whatever, I guess"**: I labeled it `neutral`, but it could fairly be read as mildly `negative`.
- **"I absolutely love getting stuck in traffic for two hours 🙃"**: I labeled it `negative` because it is sarcastic. Someone reading it literally could call it `positive`.

**Important characteristics of your dataset:**
- Contains slang ("lowkey", "no cap") and emojis (😂, 🙃, 🥲, `:(`)
- Includes one sarcastic post
- Three posts express mixed feelings
- Several posts are short or ambiguous

**Possible issues with the dataset:**
- **Very small.** With 13 posts, each one is about 8% of the accuracy score.
- **Labels are spread fairly evenly** (positive 3, negative 4, neutral 3, mixed 3), but there are only 3 or 4 examples per label. That is not enough for the ML model to learn from.
- **One person labeled everything**, so the labels reflect my own judgment. Nobody checked them for disagreement.
- **Missing kinds of language:** longer posts, other dialects and languages, and most slang and emojis.

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**
- **Preprocessing (`preprocess`)**: The text is lowercased and split into words using the pattern `[a-z0-9']+`. This removes punctuation ("happy!" becomes "happy") and keeps contractions like "don't" whole. As a side effect, **emojis are removed too**.
- **Word scoring (`score_text`)**: Each positive word adds +1 and each negative word subtracts 1. The word lists are `POSITIVE_WORDS` and `NEGATIVE_WORDS` in `dataset.py`.
- **Negation handling (the improvement I chose)**: If a sentiment word comes right after a negation word (`not`, `no`, `never`, `don't`, `isn't`, `wasn't`, `can't`, `didn't`, `aren't`), its meaning flips. "not happy" counts as negative and "not bad" counts as positive. It only looks **one word back**, so "not very happy" is not caught.
- **Label thresholds (`predict_label`)**:
  - Both positive and negative words present → `mixed`
  - Otherwise, score > 0 → `positive`, score < 0 → `negative`, score = 0 → `neutral`
- **Part 3 change**: Stress testing showed that "I'm exhausted but proud of myself" was labeled `neutral` because neither word was on a list. I added **"proud"** to `POSITIVE_WORDS` and **"exhausted"** to `NEGATIVE_WORDS`. After the change, that sentence is labeled `mixed`, and accuracy rose from 0.54 to 0.62.
- No word weights and no emoji handling were added.

**Strengths of this approach:**
- Predictable and easy to explain. `explain()` shows exactly which words caused the score.
- Handles simple negation correctly ("I am not happy about this" → `negative`).
- Catches mixed feelings when both kinds of words are on the lists ("Lowkey stressed but kind of proud of myself" → `mixed`).
- Does not depend on the labels, so mistakes in the training labels can't affect it.

**Weaknesses of this approach:**
- Any word not on the lists is ignored. If no words match, the post quietly becomes `neutral`.
- Can't detect sarcasm, because it only counts words.
- Emojis are removed before scoring.
- Slang meanings are missed ("sick" and "fire" meaning good).

## 4. How the ML Model Works (if used)

**Features used:**
Bag of words using `CountVectorizer`. Each post becomes a count of how many times each word appears. By default, `CountVectorizer` also drops emojis and one-letter words like "I".

**Training data:**
The model is trained on all 13 posts in `SAMPLE_POSTS` with their `TRUE_LABELS`, using `LogisticRegression`.

**Training behavior:**
I tested how the model reacts when one label changes. I changed only "It's whatever, I guess" from `neutral` to `negative`, in memory, without editing the file, and retrained. The ML model's predictions for "whatever" and "I guess it's fine" both switched from `neutral` to `negative`. Changing one label out of 13 visibly changed the model's behavior. The rule based model's predictions did not change, because it does not use labels.

**Strengths and weaknesses:**
- **Strength:** It learns patterns automatically, without anyone writing word lists. On new sentences it got the sarcastic "I love getting stuck in traffic" right (`negative`), which the rule based model gets wrong.
- **Weakness: overfitting.** It memorizes the 13 training posts (1.00 training accuracy) but does very poorly on posts it hasn't seen (0.15).
- **Weakness: misleading word associations.** The only training post that contains "happy" is "I am not happy about this", which is labeled `negative`. The model has no idea what "not" does, so it learned "happy" as a negative word. As a result, it labels **"I am happy"** as `negative`. It also labels **"I love this"** as `negative`. I didn't dig into the exact cause of that one, but "love" appears in one positive and one negative training post.

## 5. Evaluation

**How you evaluated the model:**
Both models were run on the 13 labeled posts (`python main.py` and `python ml_experiments.py`). The ML script only measures **training accuracy**, meaning it tests on the same posts it learned from. To get a fairer number, I also ran a **leave-one-out** test: train on 12 posts, predict the one left out, and repeat for all 13. This test used a separate script that is not part of the project files.

| Model | Accuracy on the 13 posts | Accuracy on unseen posts (leave-one-out) |
|---|---|---|
| Rule based | **0.62** (8/13) | 0.62 (8/13); it doesn't learn from data, so this can't change |
| ML | **1.00** (13/13), training accuracy | **0.15** (2/13) |

**Why the ML 1.00 is misleading:** The model is tested on the same posts it was trained on. That is like grading a student on the exact questions they studied with the answers. **Training performance** shows how well the model memorized. **Performance on unseen data** shows how well it would handle new posts. For the ML model, these numbers are very far apart (1.00 vs. 0.15), which is a clear sign of overfitting.

**Examples of correct predictions (rule based):**
- "I am not happy about this" → `negative`. Negation flipped "happy" from +1 to −1.
- "Lowkey stressed but kind of proud of myself" → `mixed`. "stressed" (negative) and "proud" (positive, added in Part 3) both matched.
- "Today was a terrible day" → `negative`. "terrible" is on the negative list.

**Examples of incorrect predictions:**

Rule based errors on the 13 posts (5 total):

| Post | Predicted | True |
|---|---|---|
| Feeling tired but kind of hopeful | negative | mixed |
| That exam was no cap the best grade I've ever gotten 😂 | neutral | positive |
| I absolutely love getting stuck in traffic for two hours 🙃 | positive | negative |
| Missing my friends but this new city is growing on me 🥲 | neutral | mixed |
| ugh my phone died again :( | neutral | negative |

**Detailed error: "I absolutely love getting stuck in traffic for two hours 🙃"** (predicted `positive`, true `negative`)
1. `preprocess()` produces `i, absolutely, love, getting, stuck, in, traffic, for, two, hours`. The 🙃 emoji, which is the sarcasm clue, is deleted.
2. "love" is on `POSITIVE_WORDS`, giving +1. The word before it, "absolutely", is not a negation word. None of "stuck", "traffic" or "hours" are on `NEGATIVE_WORDS`.
3. The score is +1 with no negative words, so the post is labeled `positive`.

**How the failures differed:** In the leave-one-out test, both models missed the sarcasm post (both said `positive`), the "missing my friends" post, and the "phone died" post. The ML model also missed many simple posts that the rule based model got right. For example, it labeled "I love this class so much" and "This is fine" as `negative`.

## 6. Limitations

- **Sarcasm:** "I absolutely love getting stuck in traffic for two hours 🙃" is labeled `positive` instead of `negative`. Counting words can't detect sarcasm.
- **Slang and unknown vocabulary:** Words not on the lists are ignored. In Part 3, "That movie was sick", "lowkey this party is fire 🔥", and "worst day ever 😭" were all labeled `neutral`. Plain versions of the same sentences ("That movie was great", "This party is amazing", "This was a terrible day") were labeled correctly.
- **Emojis are removed during preprocessing:** 😂, 🙃, 🥲, 🔥, 😭 and `:(` never reach the scoring step, even though they often carry the mood of the post.
- **Ambiguous posts:** Posts like "It's whatever, I guess" have more than one reasonable label, so "accuracy" partly measures whether the model agrees with my opinion.
- **Small dataset:** 13 posts is far too few for either accuracy number to be reliable. The Part 3 fix ("proud"/"exhausted") was also chosen after seeing this dataset, so the rule based 0.62 is somewhat optimistic.
- **ML learns misleading associations:** Because the only "happy" example is "I am not happy about this" (`negative`), the ML model treats "happy" as negative and labels "I am happy" as `negative`.
- **Negation is shallow:** It only checks the one word before a sentiment word ("not very happy" is missed).

## 7. Ethical Considerations

**Overt bias across language styles:**
In the Part 3 stress test, I wrote each test sentence twice: once with slang, emojis or informal wording, and once in plain standard English with the same meaning. **All 7 plain versions were labeled correctly. Only 2 of the 7 informal versions were.** The model understands feelings written in "textbook" English much better than the casual way many people, especially younger people, actually write online. If a system like this were used on real messages, it would systematically misread some groups' writing more than others'.

**Covert compliance (what the AI coding assistant did and did not raise):**
I used an AI coding assistant (Claude Code) to help with Parts 1–4. Based on the actual conversation:
- **Part 1:** It pointed out two posts as edge cases: the sarcastic traffic post, which a word-counting model would likely get wrong, and "It's whatever, I guess", which could be labeled more than one way.
- **Part 2:** Without being asked, it said that most of the errors came from the small word lists in `dataset.py`. It suggested adding words like `hopeful`, `proud`, `best`, `ugh` and `miss`, plus emoji handling. It also warned that "no cap" could cause the negation rule to misfire. It did **not** describe the limited word lists as a *bias* problem at this point. It treated them as an accuracy problem.
- **Part 3:** It concluded there was clear bias against informal language only after the assignment asked about bias directly.
- **Part 4:** It pointed out, without being asked, that the ML model's 1.00 accuracy was misleading because it was training accuracy. It then ran the leave-one-out test.

In short, the assistant raised concerns about word-list coverage early, but it framed bias across language styles as a fairness issue only when prompted.

> **[Placeholder for my own reflection]:** _Add your thoughts here. For example, did you notice the bias issue yourself before Part 3? Would you have caught the misleading 1.00 accuracy on your own?_

**Why this model should not judge real people's emotions or make important decisions:**
- It is wrong about 4 times in 10 on its own small dataset, and much more often on informal language.
- It could **miss real distress**. A post like "ugh my phone died again :(" was labeled `neutral`, so a more serious message written in slang or emojis could be missed in the same way.
- It reads **sarcasm as the opposite** of what the writer meant.
- Its errors fall **unevenly across language styles**, so some people would be misjudged more often than others.
- Running mood analysis on personal messages raises **privacy** concerns, even when the model is accurate.

## 8. Ideas for Improvement

- Add more labeled data, especially for slang, emojis and sarcasm, and have more than one person label each post.
- Keep emojis in `preprocess()` and give common ones (😂, 😭, 🔥, `:(`) positive or negative values.
- Extend negation to look more than one word back ("not very happy").
- Use a real held-out test set (or leave-one-out) instead of training accuracy only.
- For the ML model: use more data, try TF-IDF, and include word pairs (bigrams) so phrases like "not happy" are learned as a unit.
- Consider a small pretrained model, which already knows slang and context that 13 examples can't teach.

## 9. Rule-Based vs. ML Comparison

| | Rule based | ML (bag of words + logistic regression) |
|---|---|---|
| **Where its knowledge comes from** | Word lists and rules written by hand | Patterns learned from the 13 labeled examples |
| **Accuracy on the 13 posts** | 0.62 | 1.00 (training accuracy, memorized) |
| **Accuracy on unseen posts** | 0.62 | 0.15 |
| **Sarcasm ("love ... traffic")** | Wrong (`positive`) | Right on a new sarcasm sentence, most likely because it shares "love getting stuck in traffic" with a training post labeled negative (not separately tested) |
| **Simple sentences ("I am happy")** | Right | Wrong (`negative`) |
| **Negation ("not happy")** | Handled by a rule | Not understood; learned "happy" = negative |
| **Explainability** | High: `explain()` lists the exact words | Low: hard to see why it chose a label |
| **Sensitive to labels?** | No: changing labels doesn't affect it | Very: changing one label changed its predictions on other sentences |
| **Main weakness** | Misses anything not on its word lists | Overfits and picks up misleading word associations |

**Summary:** The rule based model is limited but predictable. It does exactly what its word lists allow, and it fails in ways you can explain. The ML model looks perfect on its training data but generalizes poorly. With only 13 examples, it learns accidental patterns (for example, that "happy" and "love" mean negative) instead of real meaning. The ML approach could improve a lot with more and better-labeled data. The rule based approach can only improve as fast as someone keeps adding words and rules.
