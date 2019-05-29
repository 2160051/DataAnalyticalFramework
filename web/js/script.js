var csvfile = null

function showTable(){
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


function clickImported(){
  var file = document.getElementById("file").files[0];
  Papa.parse(file, {
    complete: function(results) {
        eel.csvUpload(results.data)(function(ret){
          document.getElementById("csvTable").innerHTML = ret;
          return ret
        });
        var contents = document.getElementsByClassName("tableShow");
        var i;
        for (i = 0; i < contents.length; i++) {
          contents[i].style.display = "block";
        }
    }
  });
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
  console.log(regressionICount);
  if(regressionICount > 1 && regressionDCount > 1){
    alert("You can only select one dependent variable and one independent variable.")
  } else if(regressionICount > 1){
    alert("You can only select one independent variable.");
  } else if(regressionDCount > 1){
    alert("You can only select one dependent variable.");
  }
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
    eel.lin_adj_rsquare(dv, idv)(function(ret){
      document.getElementById("adjustedrsquared").innerHTML = ret;
    });
    eel.lin_pearson(dv, idv)(function(ret){
      document.getElementById("pearson").innerHTML = ret;
    });
    //eel.lin_regression(dv, idv)(function(ret){
    //  document.getElementById("display").srcdoc = ret;
    //});
    eel.lin_regression(dv, idv)(function(ret){
      document.getElementById("display").srcdoc = ret;
    });
  }else if(document.getElementById("polynomialRegression").checked) {
    document.getElementById("regression-rightbar-content").style.display = "block";
    document.getElementById("polynomial-regression-leftbar-content").style.display = "block";
    document.getElementById("linear-regression-leftbar-content").style.display = "none";
    document.getElementById("regressionLabel").innerHTML = "Polynomial Regression";
    eel.poly_int(dv, idv)(function(ret){
      document.getElementById("intercept").innerHTML = ret;
    });
    eel.poly_coefficient(dv, idv)(function(ret){
      document.getElementById("polyCoefficient").innerHTML = ret;
    });
    eel.poly_rsquared(dv, idv)(function(ret){
      document.getElementById("polyrsquared").innerHTML = ret;
    });
    eel.poly_pearson_r(dv, idv)(function(ret){
      document.getElementById("polypearson").innerHTML = ret;
    });
    eel.poly_equation(dv, idv)(function(ret){
      document.getElementById("polyequation").innerHTML = ret;
    });
  }
}

function naiveBayesResultsDisplay(){
  var naiveBayesTCount = document.getElementById("naiveBayesTarget").childElementCount;
  if(naiveBayesTCount > 1){
    alert("You can only select one target feature.");
  }
  document.getElementById("regression-rightbar-content").style.display = "none";
  document.getElementById("linear-regression-leftbar-content").style.display = "none";
  document.getElementById("polynomial-regression-leftbar-content").style.display = "none";
  document.getElementById("naiveBayes-rightbar-content").style.display = "block";
  document.getElementById("naiveBayes-leftbar-content").style.display = "block";
  document.getElementById("kmeans-rightbar-content").style.display = "none";
  document.getElementById("kmeans-leftbar-content").style.display = "none";
  document.getElementById("regressionLabel").innerHTML = "Regression";
  var ny = document.getElementById("naiveBayesTarget").getElementsByTagName("span")[0].innerHTML;
  var nX = [];
  var xs = document.getElementById("naiveBayesFeatures");
  var nxs = xs.getElementsByTagName("span");
  for(i=0;i<nxs.length;i++){
    nX.push(nxs[i].innerHTML);
  }
  eel.naive_classify(nX, y)(function(ret){
    document.getElementById("naiveResult").innerHTML = ret;
  });
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
  eel.kmeans_centroids(kdf, c)(function(ret){
    document.getElementById("centroidValues").innerHTML = ret;
  });
  eel.kmeans_sil_coef(kdf, c)(function(ret){
    document.getElementById("silhouetteCoefficient").innerHTML = ret;
  });
  document.getElementById("centroidButton").style.backgroundColor = "#E3E6E6";
  eel.kmeans_centroid_chart(kdf, c)(function(ret){
    document.getElementById("display").srcdoc = ret;
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
