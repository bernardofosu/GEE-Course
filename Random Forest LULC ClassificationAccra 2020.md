# 🌳 Random Forest LULC Classification: Accra 2020

Your Google Earth Engine (GEE) 🌍 script uses **Random Forest**, a supervised machine-learning algorithm, to classify land cover in **Accra, Ghana, for 2020**.

It produces a map with four classes:

- 💧 Water
- 🌿 Vegetation
- 🟡 Bareland
- 🏙️ Built-up

The model learns from your **labelled training points**, extracts **Sentinel-2 spectral values** at those locations, learns the patterns of each class, and then predicts the class of every other pixel across Accra.

Let's go through the code section by section, using simple examples. 👇

---

## 1. 🗺️ The Complete Workflow

| Step | What happens |
|:---:|---|
| 📍 1 | **Labelled training points:** water, vegetation, bareland and built-up |
| 🛰️ 2 | **Sentinel-2 imagery for 2020** |
| ☁️ 3 | **Cloud masking and median composite** |
| 🎨 4 | **Extract spectral features:** B2, B3, B4, B8, B11 and B12 |
| 🌳 5 | **Train Random Forest:** 100 decision trees |
| ✅ 6 | **Classify and validate:** land-cover map and accuracy assessment |
| 💾 7 | **Export** the final map to Google Drive |

---

## 2. 📂 Section 1: Load Your Uploaded Assets

Your code starts with:

```javascript
var water = ee.FeatureCollection(
  'projects/seraphic-music-452922-a5/assets/Water'
);
```

It does the same for vegetation, bareland, built-up and your Accra boundary.

### ❓ What is an asset?

An **asset** is a dataset stored in your Google Earth Engine account or project.

Your four class assets contain the reference locations you prepared for training. In this script, they are expected to be points or other geometries that represent each land-cover class.

### 📐 The area of interest (AOI)

```javascript
var aoi = accra.geometry();

Map.centerObject(accra, 9);
```

- `accra.geometry()` extracts the boundary geometry.
- `Map.centerObject(accra, 9)` zooms the map to Accra.
- `Map.addLayer(...)` displays the boundary, initially hidden because its final argument is `false`.

> 💡 **Why this matters:** the boundary tells GEE where to collect imagery and where to produce the final land-cover map.

---

## 3. 🔢 Section 2: Assign Numeric Class Labels

| Class label | Land-cover class | Map colour |
|:---:|---|---|
| 0 | 💧 Water | 🔵 Blue |
| 1 | 🌿 Vegetation | 🟢 Green |
| 2 | 🟡 Bareland | 🟡 Yellow |
| 3 | 🏙️ Built-up | 🔴 Red |

For example:

```javascript
water = water.map(function(f) {
  return f.set('class', 0);
});
```

This means: go through every water feature and add a property called `class` with the value `0`. The other datasets receive their own labels.

### 🤔 Why use numbers?

The classifier needs a target property to learn. Instead of the text label `"Water"`, it learns the numeric value `0`.

> ⚠️ The class number is only an **identifier**. It does not mean water is less important than built-up land.

### 🔗 Combine the training points

```javascript
var allPoints = water
  .merge(vegetation)
  .merge(bareland)
  .merge(builtup);

allPoints = allPoints.filterBounds(aoi);
```

- `.merge()` combines the four feature collections into one.
- `.filterBounds(aoi)` keeps features that intersect the Accra boundary.

You now have **one collection** with labelled examples from all four classes. ✅

> 📝 **Note:** the `print()` statements report the class counts *before* the combined collection is filtered to Accra, so those counts may include features outside the boundary.

---

## 4. 🛰️ Section 3: Load Sentinel-2 Imagery for 2020

```javascript
var s2 = ee.ImageCollection(
  'COPERNICUS/S2_SR_HARMONIZED'
)
.filterBounds(aoi)
.filterDate('2020-01-01', '2021-01-01')
.filter(
  ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)
);
```

| Part | What it does |
|---|---|
| `ImageCollection` | 🛰️ Loads a collection of Sentinel-2 images |
| `filterBounds(aoi)` | 📍 Keeps images that intersect Accra |
| `filterDate(...)` | 📅 Selects images from 1 January 2020 up to (not including) 1 January 2021 |
| `CLOUDY_PIXEL_PERCENTAGE < 70` | ☁️ Excludes scenes with 70% or more reported cloud cover |

> ⚠️ **Important:** the cloud percentage applies to the **whole scene**, not just the part over Accra. A cloudy scene can still be clear over your study area.

### ☁️ What does the cloud-masking function do?

```javascript
function maskS2(image) {
  var scl = image.select('SCL');

  var mask = scl.neq(1)
    .and(scl.neq(3))
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));

  return image.updateMask(mask);
}
```

The `SCL` band is the Sentinel-2 **Scene Classification Layer**. Your function removes these categories:

| SCL value | Meaning |
|:---:|---|
| 1 | 🚫 Saturated or defective pixels |
| 3 | 🌑 Cloud shadows |
| 8 | ☁️ Medium-probability clouds |
| 9 | ☁️☁️ High-probability clouds |
| 10 | 🌫️ Cirrus |
| 11 | ❄️ Snow or ice |

In simple terms, it tries to remove contaminated pixels before the model uses the imagery. It does **not** guarantee that every cloud or shadow is removed.

### 🧮 What is the median composite?

```javascript
var composite = s2
  .map(maskS2)
  .median()
  .clip(aoi);
```

For each pixel location, GEE takes the **median** of all valid observations in each band.

> 🧪 **Example:** a pixel has three valid values: `100, 120, 900`. The median is **120**, which is far less affected by the unusual 900 than the mean would be.

- The composite combines observations across 2020; it is **not** a single image from one date.
- `clip(aoi)` limits the image to the Accra boundary. ✂️

---

## 5. 🎨 Section 4: Select the Predictor Bands

```javascript
var bands = [
  'B2', 'B3', 'B4',
  'B8', 'B11', 'B12'
];

var image = composite.select(bands);
```

These bands become the **features (predictors)** that Random Forest learns from.

| Band | Name | What it measures |
|:---:|---|---|
| B2 | 🔵 Blue | Reflected blue light |
| B3 | 🟢 Green | Reflected green light |
| B4 | 🔴 Red | Reflected red light |
| B8 | 🌿 Near-infrared (NIR) | Near-infrared reflectance |
| B11 | 🔥 Shortwave infrared (SWIR 1) | Shortwave infrared reflectance |
| B12 | 🔥 Shortwave infrared (SWIR 2) | Shortwave infrared reflectance |

Different surfaces reflect light differently:

- 🌿 Healthy vegetation often reflects **strongly in NIR**.
- 💧 Water commonly **absorbs** much of the NIR.
- 🏙️🟡 Built-up surfaces and bare soil can have **overlapping** spectral patterns.

Random Forest learns combinations of these band values that separate your four classes.

> 📏 **Resolution note:** B2, B3, B4 and B8 are 10 m bands, while B11 and B12 are 20 m bands. Sampling at 10 m aligns them, but the SWIR bands do not gain any extra spatial detail.

---

## 6. 📍 Section 5: Extract Spectral Values at the Training Points

```javascript
var samples = image.sampleRegions({
  collection: allPoints,
  properties: ['class'],
  scale: 10,
  geometries: true,
  tileScale: 4
});
```

This is where your labelled locations are **connected to the satellite imagery**. 🔗

For each training point, GEE extracts the six band values and keeps the `class` property. One vegetation point might look like this:

| Feature | Illustrative value |
|---|---:|
| B2 | 450 |
| B3 | 700 |
| B4 | 350 |
| B8 | 3,200 |
| B11 | 1,100 |
| B12 | 650 |
| `class` | **1** 🌿 |

> 🧪 These numbers are **illustrative**, not values from your data.

The model receives the six band values as **inputs** and `class = 1` as the **known answer**.

- `geometries: true` keeps the sample locations.
- `tileScale: 4` helps manage memory use.

### 🧹 Remove incomplete samples

```javascript
samples = samples.filter(
  ee.Filter.notNull(bands.concat(['class']))
);
```

This removes samples with a missing band value or class. Every usable sample now has **input features + a known label**. ✅

---

## 7. ✂️ Section 6: Split into Training and Validation Sets

```javascript
var randomSamples = samples.randomColumn('random', 42);

var training = randomSamples.filter(
  ee.Filter.lt('random', 0.7)
);

var validation = randomSamples.filter(
  ee.Filter.gte('random', 0.7)
);
```

| Set | Rule | Share |
|---|---|:---:|
| 📚 Training | random value below 0.7 | about 70% |
| 🧪 Validation | random value 0.7 or higher | about 30% |

🎲 The `42` is the **random seed**. It makes the split reproducible when the input data stay the same.

### 🤔 Why split the data?

- 📚 **Training samples** teach the model.
- 🧪 **Validation samples** check how well the model predicts samples it has **never seen**.

> 🧑‍🏫 Like a teacher who does not test students with homework questions they have already seen.

> ⚠️ **Limitations:**
> - A random split can put nearby, similar samples in both sets, so accuracy may look **better than it really is** in a new area. 🗺️ Spatially separated validation is more informative.
> - The 70/30 split is not forced to be balanced within each class. Check the class counts in the Console. 🔍

---

## 8. 🌳 Section 7: Train the Random Forest

This is the core machine-learning step:

```javascript
var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: 100,
  seed: 42
}).train({
  features: training,
  classProperty: 'class',
  inputProperties: bands
});
```

| Part | Meaning |
|---|---|
| `smileRandomForest` | 🌳 Creates a Random Forest classifier |
| `numberOfTrees: 100` | 🌲 Builds 100 decision trees |
| `features: training` | 📚 Supplies the training samples |
| `classProperty: 'class'` | 🎯 The property holding the known answer |
| `inputProperties: bands` | 🎨 Use B2, B3, B4, B8, B11 and B12 as predictors |

### 🗳️ How do the 100 trees work?

1. 🎲 Each tree learns from a **random bootstrap sample** of the training points.
2. 🔀 At each split, it only considers a **random subset** of the bands. This makes the trees different from each other.
3. 🌿 Different trees learn different rules, for example combinations of red, NIR and SWIR that separate vegetation from built-up.
4. 🗳️ For a new pixel, every tree votes for a class, and the **majority vote** wins.

> 🧪 **Example:** 65 of 100 trees vote vegetation, so the pixel becomes **vegetation**. 🌿

`rf.explain()` prints information about the trained model in the Console. 🖨️

---

## 9. 🗺️ Section 8: Classify the Whole Image

```javascript
var classified = image.classify(rf).rename('LULC');
```

The trained Random Forest is applied to every pixel, which gets one of four labels:

- `0`: 💧 Water
- `1`: 🌿 Vegetation
- `2`: 🟡 Bareland
- `3`: 🏙️ Built-up

The result is a land-use/land-cover image called `LULC`.

```javascript
var palette = [
  '1565C0',  // 🔵 blue: water
  '238B45',  // 🟢 green: vegetation
  'E6C84F',  // 🟡 yellow: bareland
  'D73027'   // 🔴 red: built-up
];
```

> 🎨 These are only **display colours**. The pixel values remain the numeric class labels.

---

## 10. ✅ Section 9: Validate the Classification

```javascript
var validated = validation.classify(rf);

var confusionMatrix = validated.errorMatrix(
  'class',
  'classification',
  [0, 1, 2, 3]
);
```

The confusion matrix compares the **reference labels** (`class`) with the **predictions** (`classification`). It shows, for example, how many vegetation samples were correctly classified and how many were confused with water, bareland or built-up.

| Output | Meaning |
|---|---|
| 📊 Confusion matrix | Counts correct and incorrect class predictions |
| 🎯 Overall accuracy | Share of validation samples classified correctly |
| 🤝 Kappa coefficient | Agreement adjusted for chance |
| 🌿 Producer's accuracy | How well reference samples of each class were detected |
| 🔍 Consumer's accuracy | How reliable the predicted labels are for each class (also called user's accuracy) |

$$
\text{Overall Accuracy} = \frac{\text{Correct predictions}}{\text{All validation samples}}
$$

> 🧪 **Example:** 270 of 300 validation samples correct gives **90%**. This is an example only; your real result appears in the GEE Console.

- 🌿 Producer's accuracy reveals **omission errors** (missed pixels).
- 🔍 Consumer's accuracy reveals **commission errors** (wrongly added pixels).

---

## 11. 💾 Section 10: Export the Final Map

```javascript
Export.image.toDrive({
  image: classified.toByte(),
  description: 'Accra_RF_LULC_2020',
  folder: 'GEE_Exports',
  fileNamePrefix: 'Accra_RF_LULC_2020',
  region: aoi,
  scale: 10,
  maxPixels: 1e13
});
```

| Part | Meaning |
|---|---|
| `classified.toByte()` | 🗜️ Stores class labels as small integers |
| `description`, `fileNamePrefix` | 🏷️ Name the export |
| `folder` | 📁 Google Drive folder |
| `region: aoi` | 📍 Export the Accra area |
| `scale: 10` | 📏 10 m output pixels |
| `maxPixels` | 🔢 Allows a large number of pixels |

---
---

# 🌍 Random Forest Model Evaluation: Ecosystem Mapping

## 1. 🎯 The Four Possible Outcomes

We use **vegetation** 🌿 as the class we want to detect. All other classes (water, bareland and built-up) count as **not vegetation**.

### 🟢 True Positive (TP): Correct Detection
- **Actual:** Vegetation
- **Predicted:** Vegetation
- The pixel really is vegetation, and the model correctly finds it.

### 🟢 True Negative (TN): Correct Rejection
- **Actual:** Not vegetation
- **Predicted:** Not vegetation
- The pixel is not vegetation, and the model correctly rejects it.

### 🔴 False Positive (FP): False Alarm
- **Actual:** Not vegetation
- **Predicted:** Vegetation
- The model wrongly labels water, bareland or built-up as vegetation.
- This is a **commission error** for vegetation.

### 🔴 False Negative (FN): Missed Detection
- **Actual:** Vegetation
- **Predicted:** Not vegetation
- The pixel really is vegetation, but the model labels it as another class.
- This is an **omission error** for vegetation.

> 🧠 **Remember:**
> - ✅ **True** = the prediction is correct.
> - ❌ **False** = the prediction is incorrect.
> - 🎯 **Positive** = the class being detected (vegetation).
> - 🚫 **Negative** = everything outside that class.

For a four-class map, calculate these outcomes **separately for each class**, treating one class as positive and the rest as negative.

---

## 2. 📊 Confusion Matrix Example

Suppose we validate 100 reference pixels for vegetation versus not vegetation (illustrative numbers):

| Actual class | Predicted vegetation | Predicted not vegetation | Total |
|---|---:|---:|---:|
| 🌿 Vegetation | ✅ 40 (TP) | ❌ 10 (FN) | 50 |
| 🏙️ Not vegetation | ❌ 5 (FP) | ✅ 45 (TN) | 50 |
| **Total** | **45** | **55** | **100** |

- ✅ 40 vegetation pixels were correctly detected.
- ✅ 45 non-vegetation pixels were correctly rejected.
- ❌ 5 non-vegetation pixels were wrongly labelled vegetation.
- ❌ 10 vegetation pixels were missed.

---

## 3. 📈 Evaluation Metrics and Formulas

### 🎯 1. Overall Accuracy

**Question:** How many predictions were correct out of all validation pixels?

$$
\text{Overall Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Example:** $\frac{40 + 45}{100} = 85\%$

👉 The model correctly classified **85%** of the 100 validation pixels.

### 🌿 2. Sensitivity / Recall / Producer's Accuracy

**Question:** Of all the pixels that really are vegetation, how many did the model find?

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**Example:** $\frac{40}{40 + 10} = 80\%$

👉 The model found **80%** of the real vegetation and missed 20%.

- ⬆️ High recall = few vegetation pixels are missed.
- ⬇️ Low recall = many vegetation pixels are missed.

### 🛡️ 3. Specificity

**Question:** Of all the pixels that are not vegetation, how many were correctly rejected?

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

**Example:** $\frac{45}{45 + 5} = 90\%$

👉 The model correctly rejected **90%** of the non-vegetation pixels.

### 🔍 4. Precision / Consumer's Accuracy / User's Accuracy

**Question:** Of all the pixels predicted as vegetation, how many really are vegetation?

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Example:** $\frac{40}{40 + 5} \approx 88.9\%$

👉 About **88.9%** of the pixels the model called vegetation really are vegetation.

- ⬆️ High precision = few false alarms.
- ⬇️ Low precision = many "vegetation" pixels belong to another class.

### ⚖️ 5. F1-score

**Question:** How well does the model balance precision and recall?

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Example:** $2 \times \frac{0.889 \times 0.80}{0.889 + 0.80} \approx 84.2\%$

👉 One number that balances false alarms and missed detections.

### 🤝 6. Kappa Coefficient

**Question:** How much agreement is there after removing the agreement expected by chance?

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

- $p_o$ = observed agreement (overall accuracy) = 0.85
- $p_e$ = agreement expected by chance = $(0.45 \times 0.50) + (0.55 \times 0.50) = 0.50$

**Example:** $\kappa = \frac{0.85 - 0.50}{1 - 0.50} = 0.70$

👉 The model is **70%** of the way from random guessing to a perfect map. Read kappa together with overall accuracy and the class-specific metrics, not on its own.

### 🧩 7. Confusion Matrix

**Question:** Which land-cover classes are being confused?

It compares reference labels with predicted labels, for example showing whether vegetation is often mistaken for built-up or bareland. It is a **table of counts**, not a single score.

---

## 4. 🧠 Quick Comparison Table

| Metric | Simple meaning |
|---|---|
| 📊 Confusion matrix | Which classes were predicted correctly or wrongly? |
| 🎯 Overall accuracy | What share of all predictions were correct? |
| 🌿 Recall / Producer's accuracy | How much of the real class did we find? |
| 🛡️ Specificity | How well did we reject pixels outside the class? |
| 🔍 Precision / Consumer's accuracy | How trustworthy are the pixels predicted as this class? |
| ⚖️ F1-score | How well are precision and recall balanced? |
| 🤝 Kappa | How much agreement remains after removing chance? |

---

## 5. 🌍 Applying These Metrics to Accra LULC

Your model classifies four classes:

- 💧 Water
- 🌿 Vegetation
- 🟡 Bareland
- 🏙️ Built-up

Calculate producer's accuracy, consumer's accuracy, recall, precision and F1-score **for each class separately**. For example, when evaluating 🏙️ built-up, treat built-up as positive and water, vegetation and bareland as negative.

> ⚠️ **Important:** these metrics describe performance on the **validation samples**. A random split can overestimate how well the map works in new locations if nearby or similar samples appear in both sets. 🗺️ Spatially separated validation helps test this.

---

## ✅ Key Takeaways

| | Question it answers |
|---|---|
| 🌿 **Recall** | How much of the real class did we find? |
| 🔍 **Precision** | How many of our predicted pixels were correct? |
| 🛡️ **Specificity** | How well did we reject the other classes? |
| ⚖️ **F1-score** | How well did we balance precision and recall? |
| 🎯 **Overall accuracy** | How many predictions were correct overall? |
| 🤝 **Kappa** | How much agreement is there after removing chance? |
| 📊 **Confusion matrix** | Which classes are being confused? |