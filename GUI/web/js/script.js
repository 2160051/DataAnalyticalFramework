eel.expose(kmeansfunction);
function kmeansfunction(a, b, c, d) {
  if (a < b) {
    console.log(c * d);
  }
}

x = null

function toServer(x){
  if(x==results){
  eel.test()(function(ret){ret})
  }else if(x==interpret){
  }
}

function regressionDisplay(){
}

function naiveBayesDisplay(){
}

function kmeansResultsDisplay(){
  document.getElementById("kmeans-rightbar-content").style.display = "block";
  document.getElementById("kmeans-leftbar-content").style.display = "block";
  document.getElementById("centroidValues").innerHTML = " [24, 82, 98, 4], [16, 56, 76, 2]";
  document.getElementById("silhouetteCoefficient").innerHTML = " 0.6248";
  document.getElementById("clusterButton").style.backgroundColor = "#E3E6E6";
  document.getElementById("graphTemp").src = "img/clusterGraph.png";
}

function kmeansDisplay(){
  document.getElementById("kmeansButton").className = "processButtonSelected";
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