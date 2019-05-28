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
  document.getElementById("kmeans-rightbar-content").style.display = "none";
  document.getElementById("kmeans-leftbar-content").style.display = "none";
  document.getElementById("naiveBayes-rightbar-content").style.display = "none";
  document.getElementById("naiveBayes-leftbar-content").style.display = "none";
  document.getElementById("graphTemp").style.display = "none";
  var dv = document.getElementById("regressionDependent").getElementsByTagName("span")[0].innerHTML;
  var idv = document.getElementById("regressionIndependent").getElementsByTagName("span")[0].innerHTML;
  document.getElementById("lineGraphButton").style.backgroundColor = "#E3E6E6";
  if(document.getElementById("linearRegression").checked) {
    document.getElementById("regression-rightbar-content").style.display = "block";
    document.getElementById("linear-regression-leftbar-content").style.display = "block";
    document.getElementById("regressionLabel").innerHTML = "Linear Regression";
    document.getElementById("polynomial-regression-leftbar-content").style.display = "none";
    eel.lin_num_rsquare(dv, idv)(function(ret){
      document.getElementById("rsquared").innerHTML = ret;
    });
    eel.lin_adj_rsquare(dv, idv)(function(reta){
      document.getElementById("adjustedrsquared").innerHTML = reta;
    });
    eel.lin_pearson(dv, idv)(function(retp){
      document.getElementById("pearson").innerHTML = retp;
    });
    eel.lin_regression(dv, idv)(function(rets){
      document.getElementById("display").srcdoc = rets;
    });
  }else if(document.getElementById("polynomialRegression").checked) {
    document.getElementById("regression-rightbar-content").style.display = "block";
    document.getElementById("polynomial-regression-leftbar-content").style.display = "block";
    document.getElementById("linear-regression-leftbar-content").style.display = "none";
    document.getElementById("regressionLabel").innerHTML = "Polynomial Regression";
    eel.poly_int(dv, idv)(function(ret){
      document.getElementById("intercept").innerHTML = ret;
    });
    eel.poly_coefficient(dv, idv)(function(retc){
      document.getElementById("polyCoefficient").innerHTML = retc;
    });
    eel.poly_rsquared(dv, idv)(function(retr){
      document.getElementById("polyrsquared").innerHTML = retr;
    });
    eel.poly_pearson_r(dv, idv)(function(retp){
      document.getElementById("polypearson").innerHTML = retp;
    });
    eel.poly_equation(dv, idv)(function(rete){
      document.getElementById("polyequation").innerHTML = rete;
    });
  }
}

function naiveBayesResultsDisplay(){
  document.getElementById("regression-rightbar-content").style.display = "none";
  document.getElementById("linear-regression-leftbar-content").style.display = "none";
  document.getElementById("polynomial-regression-leftbar-content").style.display = "none";
  document.getElementById("naiveBayes-rightbar-content").style.display = "block";
  document.getElementById("naiveBayes-leftbar-content").style.display = "block";
  document.getElementById("kmeans-rightbar-content").style.display = "none";
  document.getElementById("kmeans-leftbar-content").style.display = "none";
  document.getElementById("regressionLabel").innerHTML = "Regression";
}

function kmeansResultsDisplay(){
  document.getElementById("regression-rightbar-content").style.display = "none";
  document.getElementById("linear-regression-leftbar-content").style.display = "none";
  document.getElementById("polynomial-regression-leftbar-content").style.display = "none";
  document.getElementById("naiveBayes-rightbar-content").style.display = "none";
  document.getElementById("naiveBayes-leftbar-content").style.display = "none";
  document.getElementById("kmeans-rightbar-content").style.display = "block";
  document.getElementById("kmeans-leftbar-content").style.display = "block";
  document.getElementById("graphTemp").style.display = "none";
  document.getElementById("regressionLabel").innerHTML = "Regression";
  var c = document.getElementById("clusterNumber").value;
  var kdf = [];
  var kf = document.getElementById("kmeansFeatures");
  var nkf = kf.getElementsByTagName("span");
  for(i=0;i<nkf.length;i++){
    kdf.push(nkf[i].innerHTML);
  }
  eel.kmeans_centroids(kdf, c)(function(retc){
    document.getElementById("centroidValues").innerHTML = retc;
  });
  eel.kmeans_sil_coef(kdf, c)(function(ret){
    document.getElementById("silhouetteCoefficient").innerHTML = ret;
  });
  document.getElementById("centroidButton").style.backgroundColor = "#E3E6E6";
  eel.kmeans_centroid_chart(kdf, c)(function(retv){
    document.getElementById("display").srcdoc = retv;
  });
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
