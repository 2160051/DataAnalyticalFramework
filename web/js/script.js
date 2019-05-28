var csvfile = null

function clickImported(){
  var file = document.getElementById("file").files[0];
  Papa.parse(file, {
    complete: function(results) {
        eel.csvUpload(results.data)
    }
  });
  eel.table()(function(ret){
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
  var regressionICount = document.getElementById("regressionIndependent").childElementCount;
  var regressionDCount = document.getElementById("regressionDependent").childElementCount;
  if(regressionICount > 1 && regressionDCount > 1){
    alert("You can only select one dependent variable and one independent variable.")
  } else if(regressionICount > 1){
    alert("You can only select one independent variable.");
  } else if(regressionDCount > 1){
    alert("You can only select one dependent variable.");
  }
  document.getElementById("kmeans-rightbar-content").style.display = "none";
  document.getElementById("naiveBayes-rightbar-content").style.display = "none";
  document.getElementById("graphTemp").style.display = "none";
  var dv = document.getElementById("regressionDependent").getElementsByTagName("span")[0].innerHTML;
  var idv = document.getElementById("regressionIndependent").getElementsByTagName("span")[0].innerHTML;
  document.getElementById("lineGraphButton").style.backgroundColor = "#E3E6E6";
  if(document.getElementById("linearRegression").checked) {
    document.getElementById("regression-rightbar-content").style.display = "block";
    document.getElementById("regressionLabel").innerHTML = "Linear Regression";
    eel.lin_regression(dv, idv)(function(ret){
      document.getElementById("rdisplay").srcdoc = ret;
    });
    eel.lin_rtable(dv, idv)(function(ret){
      document.getElementById("regressionTable").innerHTML = ret;
    });
  }else if(document.getElementById("polynomialRegression").checked) {
    document.getElementById("regression-rightbar-content").style.display = "block";
    document.getElementById("regressionLabel").innerHTML = "Polynomial Regression";
    eel.poly_regression(dv, idv)(function(ret){
      document.getElementById("rdisplay").srcdoc = ret;
    });
    eel.poly_rtable(dv, idv)(function(ret){
      document.getElementById("regressionTable").innerHTML = ret;
    });
  }
}

function naiveBayesResultsDisplay(){
  var naiveBayesTCount = document.getElementById("naiveBayesTarget").childElementCount;
  if(naiveBayesTCount > 1){
    alert("You can only select one target feature.");
  }
  document.getElementById("regression-rightbar-content").style.display = "none";
  document.getElementById("naiveBayes-rightbar-content").style.display = "block";
  document.getElementById("kmeans-rightbar-content").style.display = "none";
  document.getElementById("graphTemp").style.display = "none";
  document.getElementById("regressionLabel").innerHTML = "Regression";
  document.getElementById("confusionButton").style.backgroundColor = "#E3E6E6";
  var ny = document.getElementById("naiveBayesTarget").getElementsByTagName("span")[0].innerHTML;
  var nX = [];
  var xs = document.getElementById("naiveBayesFeatures");
  var nxs = xs.getElementsByTagName("span");
  for(i=0;i<nxs.length;i++){
    nX.push(nxs[i].innerHTML);
  }
  eel.naive_matrix(nX, ny)(function(ret){
    document.getElementById("naiveResult").innerHTML = ret;
  });
}

var c = '';
var kdf = [];

function kmeansResultsDisplay(){
  document.getElementById("regression-rightbar-content").style.display = "none";
  document.getElementById("naiveBayes-rightbar-content").style.display = "none";
  document.getElementById("kmeans-rightbar-content").style.display = "block";
  document.getElementById("graphTemp").style.display = "none";
  document.getElementById("regressionLabel").innerHTML = "Regression";
  c = document.getElementById("clusterNumber").value;
  var kf = document.getElementById("kmeansFeatures");
  var nkf = kf.getElementsByTagName("span");
  for(i=0;i<nkf.length;i++){
    kdf.push(nkf[i].innerHTML);
  }
  eel.kmeans_centroids(kdf, c)(function(ret){
    document.getElementById("centroidValues").innerHTML = ret;
  });
  eel.kmeans_sil_coef(kdf, c)(function(ret){
    document.getElementById("silhouetteCoefficient").innerHTML = ret;
  });
  document.getElementById("centroidButton").style.backgroundColor = "#E3E6E6";
  eel.kmeans_centroid_chart(kdf, c)(function(ret){
    console.log(ret);
    document.getElementById("kdisplay").srcdoc = ret;
  });
}

function cluster() {
  document.getElementById("clusterButton").style.backgroundColor = "#E3E6E6";
  document.getElementById("centroidButton").style.backgroundColor = "#FFFFFF";
  var c = document.getElementById("clusterNumber").value;
  var kdf = [];
  var kf = document.getElementById("kmeansFeatures");
  var nkf = kf.getElementsByTagName("span");
  for(i=0;i<nkf.length;i++){
    kdf.push(nkf[i].innerHTML);
  }
  eel.kmeans_cluster_graph(kdf, c)(function(ret){
    document.getElementById("kdisplay").srcdoc = ret;
  });
}

function centroid() {
  document.getElementById("centroidButton").style.backgroundColor = "#E3E6E6";
  document.getElementById("clusterButton").style.backgroundColor = "#FFFFFF";
  document.getElementById("graphTemp").src = "img/centroidChart.png";
  var c = document.getElementById("clusterNumber").value;
  var kdf = [];
  var kf = document.getElementById("kmeansFeatures");
  var nkf = kf.getElementsByTagName("span");
  for(i=0;i<nkf.length;i++){
    kdf.push(nkf[i].innerHTML);
  }
  eel.kmeans_centroid_chart(kdf, c)(function(ret){
    document.getElementById("kdisplay").srcdoc = ret;
  });
}

function compare() {
  document.getElementById("confusionButton").style.backgroundColor = "#E3E6E6";
  document.getElementById("graphTemp").src = "img/centroidChart.png";
}
