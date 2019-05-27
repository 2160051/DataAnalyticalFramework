
eel.expose(csvUpload);
function csvUpload(data) {

}

function clickImported(){
  var csvFile = document.getElementById("file");
  console.log(csvFile.files[0]);
  eel.csvUpload(csvFile.files[0])(print_return);
}

function createTable(){
  eel.table()(function(ret){
    console.log(ret);
    document.getElementById("csvTable").innerHTML = ret;
    return ret
  });
  var contents = document.getElementsByClassName("tableShow");
  var i;
  for (i = 0; i < contents.length; i++) {
    contents[i].style.display = "block";
  }
}

function regressionDisplay(){
  document.getElementById("regressionButton").className = "processButtonSelected";
  document.getElementById("kmeansButton").className = "processButton";
  document.getElementById("naiveBayesButton").className = "processButton";
}

function naiveBayesDisplay(){
  document.getElementById("naiveBayesButton").className = "processButtonSelected";
  document.getElementById("regressionButton").className = "processButton";
  document.getElementById("kmeansButton").className = "processButton";
}

function kmeansDisplay(){
  document.getElementById("kmeansButton").className = "processButtonSelected";
  document.getElementById("regressionButton").className = "processButton";
  document.getElementById("naiveBayesButton").className = "processButton";
}

function regressionResultsDisplay(){
  document.getElementById("regression-rightbar-content").style.display = "block";
  document.getElementById("regression-leftbar-content").style.display = "block";
  document.getElementById("kmeans-rightbar-content").style.display = "none";
  document.getElementById("kmeans-leftbar-content").style.display = "none";
  document.getElementById("naiveBayes-rightbar-content").style.display = "none";
  document.getElementById("naiveBayes-leftbar-content").style.display = "none";
}

function naiveBayesResultsDisplay(){
  document.getElementById("regression-rightbar-content").style.display = "none";
  document.getElementById("regression-leftbar-content").style.display = "none";
  document.getElementById("naiveBayes-rightbar-content").style.display = "block";
  document.getElementById("naiveBayes-leftbar-content").style.display = "block";
  document.getElementById("kmeans-rightbar-content").style.display = "none";
  document.getElementById("kmeans-leftbar-content").style.display = "none";
}

function kmeansResultsDisplay(){
  document.getElementById("regression-rightbar-content").style.display = "none";
  document.getElementById("regression-leftbar-content").style.display = "none";
  document.getElementById("naiveBayes-rightbar-content").style.display = "none";
  document.getElementById("naiveBayes-leftbar-content").style.display = "none";
  document.getElementById("kmeans-rightbar-content").style.display = "block";
  document.getElementById("kmeans-leftbar-content").style.display = "block";
  document.getElementById("centroidValues").innerHTML = " [24, 82, 98, 4], [16, 56, 76, 2]";
  document.getElementById("silhouetteCoefficient").innerHTML = " 0.6248";
  document.getElementById("clusterButton").style.backgroundColor = "#E3E6E6";
  document.getElementById("graphTemp").src = "img/clusterGraph.png";
}

function cluster() {
  document.getElementById("clusterButton").style.backgroundColor = "#E3E6E6";
  document.getElementById("centroidButton").style.backgroundColor = "#FFFFFF";
  document.getElementById("graphTemp").src = "img/clusterGraph.png";
}

function centroid() {
  document.getElementById("centroidButton").style.backgroundColor = "#E3E6E6";
  document.getElementById("clusterButton").style.backgroundColor = "#FFFFFF";
  document.getElementById("graphTemp").src = "img/centroidChart.png";
}

function compare() {
  document.getElementById("confusionButton").style.backgroundColor = "#E3E6E6";
  document.getElementById("graphTemp").src = "img/centroidChart.png";
}