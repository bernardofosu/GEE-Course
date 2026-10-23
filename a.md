                 GEOSPATIAL DATA
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   Optical          SAR/Thermal     Vector
 Sentinel/Landsat   Radar/LST      GIS layers
        │              │              │
        └──────────────┼──────────────┘
                       ↓
              DATA PREPROCESSING
       cloud masking • scaling • QA
       projection • resampling • tiling
                       ↓
              FEATURE ENGINEERING
       spectral • spatial • temporal
       terrain • morphology • context
                       ↓
             MULTIMODAL FUSION
                       ↓
              SPATIAL REPRESENTATION
        pixels → objects → grids → regions
                       ↓
                ML / DEEP LEARNING
      RF • XGBoost • CNN • Transformer
                       ↓
              SPATIAL MODELLING
       prediction • classification
       regression • clustering
       change detection • forecasting
                       ↓
                 GEOAI LAYER
       spatial reasoning • agents • LLMs
                       ↓
              UNCERTAINTY / EXPLAINABILITY
                       ↓
                 APPLICATION