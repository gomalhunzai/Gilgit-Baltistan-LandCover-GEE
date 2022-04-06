//Code for landcover analysis using composite image of sentinel-2
//Copy Rights reserved with GOMAL AMIN, EARL & AKAH
 
// selection of imgery based on predefined criteria 
Map.centerObject(GB, 7);
var collection = ee.ImageCollection('COPERNICUS/S2_SR') 
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
  .filterDate('2019-05-1' ,'2019-9-30')
  .filterBounds(GB);

// print('Filtered (by date) Image Count: ', collection.size());
// print(collection) // image collection metadata 

var medianpixels = collection.median(); 
var medianpixelsclipped = medianpixels.clip(table).divide(10000); 
                               
// // Now visualise the mosaic as a natural colour image. 
Map.addLayer(medianpixelsclipped, {bands: ['B4', 'B3', 'B2'], min: 0, max: 1, gamma: 1.5},
'Sentinel_2 mosaic',false);

var NTL = ee.ImageCollection("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS")
             .filterDate('2013-1-1' ,'2014-1-1')
             .filterBounds(GB)
             .select('cf_cvg')
             .map(function(image){return image.clip(GB)});
var median_NTL = NTL.median(); 
Map.addLayer(NTL, {bands: ['cf_cvg'], min: 0, max: 84}, 'NTL');            


//supervised_Classfication 
var landuse = Water.merge(Forest).merge(Grasses).merge(Wetland).merge(Agriculture).merge(Barrenland).merge(Buildup).merge(Glacier).merge(Snow);
print(landuse)
//get the values for all pixels in each point in the training
var points = medianpixelsclipped.sampleRegions({
  collection: landuse, 
  properties: ['landcover'],
  tileScale: 16,
  scale: 10
}).randomColumn();

var training = points.filter(ee.Filter.lt('random', 0.7));
var validation1 = points.filter(ee.Filter.gte('random', 0.3));

//please select only one var from below and comment the remaining 
var Scenario_1 = ['B1', 'B2', 'B3','B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']; //Scenario#1
var Scenario_2 = ['B2', 'B3','B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']; //Scenario#2..
var Scenario_3 = ['B2', 'B3','B4', 'B5'];  //Scenario#3
var Scenario_4 = ['B2', 'B3','B4', 'B5', 'B6', 'B7']; //Scenario#4
var Scenario_5 = ['B2', 'B3','B4', 'B8A'];  //Scenario#5
var Scenario_6 = ['B2', 'B3','B4', 'B5', 'B6', 'B7', 'B8',];  //Scenario#6
var Scenario_7 = ['B2', 'B3','B4', 'B5', 'B6', 'B7', 'B8', 'B8A']; //Scenario#7
var Scenario_8 = ['B2', 'B3','B4', 'B5', 'B6', 'B7', 'B8', 'B11']; //Scenario#8
var Scenario_9 = ['B2', 'B3','B4']; //Scenario#9
var Scenario_10 = ['B5', 'B6', 'B7', 'B11', 'B12']; //Scenario#10


// Parameters to be used in the training
var classifierConfig = {
   features: training,
   classProperty: 'class',
   inputProperties: Scenario_1
};

//// Train a RF classifier with parameters
var trained = ee.Classifier.smileRandomForest({numberOfTrees: 500,
                          seed: 0}).train(training,'landcover',Scenario_1);
                          
//// Train a lib SVM classifier with default parameters
var trained = ee.Classifier.libsvm().train(training,'landcover',Scenario_1);

//// Train a gmoMaxEnt classifier with default parameters
var trained = ee.Classifier.gmoMaxEnt().train(training,'landcover',Scenario_1);

//// Train a smileCart classifier with default parameters
var trained = ee.Classifier.smileCart().train(training,'landcover',Scenario_1);

//// Train a NaiveBayes classifier with default parameters
var trained = ee.Classifier.smileNaiveBayes({lambda: 0.000001}).train(training,'landcover',Scenario_1);

//// Train a minimumDistance classifier with default parameters
var trained = ee.Classifier.minimumDistance().train(training,'landcover',Scenario_1);


print(trained)

//explain classifer 
var dict = trained.explain();
print('Explain:',dict);

// Classify the mossaiced image
var classified = medianpixelsclipped.select(Scenario_1).classify(trained);
print(classified);
  
// Display the classification result and the input image.
Map.addLayer(classified,
             {min: 0, max: 8, palette: ['320cff','50d618','98ff00','868b41','1ab317','c6c48c','b9c1c2','1572d2','738bff']},
             'classified',false);

// Get a confusion matrix representing Training resubstitution accuracy.
var trainAccuracy = trained.confusionMatrix();

print('Resubstitution error matrix: ', trainAccuracy);//The accuracy made by the model on the training data
print('Training overall accuracy: ', trainAccuracy.accuracy());


var confusionMatrix = ee.ConfusionMatrix(validation1.classify(trained)
                      .errorMatrix({
                          actual: 'landcover',
                          predicted: 'classification'
                      }))

var OA = confusionMatrix.accuracy();
var CA = confusionMatrix.consumersAccuracy();
var Kappa = confusionMatrix.kappa();
// var Order = confusionMatrix.order();
var PA = confusionMatrix.producersAccuracy();
//PRINT RESULTS
print('Confusion Matrix', confusionMatrix)
print(OA,'validation Overall Accuracy');
print(CA,'Consumers Accuracy');
print(PA,'Producers Accuracy');
print(Kappa,'Kappa');
print(Order,'Order');


// // // export to googledrive as a tiff
Export.image.toDrive({
  image: classified,
  description: 'S2_Classified',
  fileNamePrefix: 'S2_Classified', 
  scale: 10,
  maxPixels: 1211,
  region: GB,
  folder: "GBLC_output
});

Export.table.toAsset(landuse, 'landuse points', landuse)


