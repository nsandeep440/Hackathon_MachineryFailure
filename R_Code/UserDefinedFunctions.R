
#############################################
# USER DEFINED FUNCTIONS.
#############################################

ns_checkTotalMissingValues = function(dataSet) {
  print(sum(is.na(dataSet)))
}

ns_checkColumnwiseMissingValues = function(dataSet) {
  sapply(dataSet, function(x) sum(is.na(x)))
  #print(table(is.na(dataSet)))
}

ns_showEachColumnFactorProportions = function(dataSet) {
  columnList = colnames(dataSet)
  sapply(columnList, function(x) table(dataSet[x]))
}

ns_dataStatistics = function(dataSet) {
  print('----------------- dimensions-----------------------')
  dim(dataSet)
  print('----------------- summary-----------------------')
  summary(dataSet)
  print('----------------- structure-----------------------')
  str(dataSet)
  print('----------------- missing values-----------------------')
  ns_checkTotalMissingValues(dataSet) ## 3160 missing values
  print('----------------- column wise missing values-----------------------')
  ns_checkColumnwiseMissingValues(dataSet) 
}


#############################################
# USER DEFINED FUNCTIONS ---------- END.
#############################################

split.data = function(data, p = 0.75, s = 666) {
  set.seed(s)
  index = sample(1:dim(data)[1])
  train = data[index[1:floor(dim(data)[1] * p)], ]
  test = data[index[((ceiling(dim(data)[1] * p)) + 1):dim(data)[1]], ]
  return(list(train = train, test = test)) 
}

#############################################
# Save to csv file
#############################################
saveToCSV = function(probabilityForTest, fileName, testData, firstCol) {
  file1 = data.frame(ID = firstCol)
  file1["y"] = probabilityForTest
  write.csv(file1, fileName, row.names = FALSE)
}





