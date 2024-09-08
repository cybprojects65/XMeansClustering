package it.cnr.datamining.riskassessment.data.mining;

import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class XmeansFeatureClustering {

	public static void main(String[] args) throws Exception {

		File inputFile = null; 
		List<String> columnsToTake = new ArrayList<>();
		int minElementsInCluster;
		int minClusters;
		int maxClusters;
		int maxIterations;
		//example of input:
		//"./dataset_mediterranean_sea_2017_2018_2019_2020_2021.csv" 2 3 16 100 "environment 2017_net_primary_production" "environment 2017_sea-bottom_dissolved_oxygen"
		//<input file> <min_elements_in_cluster> <minimum_n_of_clusters> <maximum_n_of_clusters> <maximum_iterations> <feature1> <feature2> ... <featuren>
		if (args!= null && args.length>0) {
			inputFile = new File(args[0]);
			
			minElementsInCluster = Integer.parseInt(args[1]);
			minClusters = Integer.parseInt(args[2]);
			maxClusters = Integer.parseInt(args[3]);
			maxIterations = Integer.parseInt(args[4]);
			
			for(int i=5;i<args.length;i++)
				columnsToTake.add(args[i]);
			
			
		}else {
			
			inputFile = new File("./dataset_mediterranean_sea_2017_2018_2019_2020_2021.csv");
			columnsToTake.add("environment 2017_net_primary_production");
			columnsToTake.add("environment 2017_sea-bottom_dissolved_oxygen");
			
			minElementsInCluster = 2;
			minClusters = 3;
			maxClusters = 16;
			maxIterations = 100;
			
		}
		
		List<String> allLines = Files.readAllLines(inputFile.toPath());
		int ncells = allLines.size() - 1;
		String header = allLines.get(0);
		String head_elements[] = header.split(",");
		List<Integer> validIdxs = new ArrayList<>();
		int headerCounter = 0;
		System.out.println("Selecting headers..");
		StringBuffer headerString = new StringBuffer();
		for (String h : head_elements) {
			if (columnsToTake.contains(h)) {
				System.out.println("Selecting header: " + h);
				validIdxs.add(headerCounter);
				headerString.append(h + ",");
			}

			headerCounter++;
		}

		System.out.println("Building matrix..");
		double[][] featureMatrix = new double[ncells][validIdxs.size()];
		int lineIndex = 1;
		int nfeatures = featureMatrix[0].length;

		for (int i = 0; i < featureMatrix.length; i++) {

			String row = allLines.get(lineIndex);
			String rowE[] = row.split(",");

			for (int j = 0; j < featureMatrix[0].length; j++) {

				featureMatrix[i][j] = Double.parseDouble(rowE[validIdxs.get(j)]);

			}
			lineIndex++;
		}

		System.out.println("Added " + (lineIndex - 1) + " rows and " + validIdxs.size() + " columns");

		double[][] allFeatureMatrix = new double[ncells][validIdxs.size() + 2];
		lineIndex = 1;
		List<Integer> allValidIdxs = new ArrayList<>(validIdxs);
		allValidIdxs.add(0, 0);
		allValidIdxs.add(1, 1);

		for (int i = 0; i < allFeatureMatrix.length; i++) {

			String row = allLines.get(lineIndex);
			String rowE[] = row.split(",");

			for (int j = 0; j < allFeatureMatrix[0].length; j++) {

				allFeatureMatrix[i][j] = Double.parseDouble(rowE[allValidIdxs.get(j)]);

			}
			lineIndex++;
		}

		System.out.println("Standardising the dataset..");
		Operations operations = new Operations();
		double[][] featureMatrix_std = operations.standardize(featureMatrix);

		File outputClusteredTable = null;
		File outputClusterStatsTable = null;
		File clusteringFile = null;

		

		System.out.println("Applying Xmeans..");

		System.setProperty("rapidminer.init.operators", "./cfg/operators.xml");
		//RapidMiner.init();
		System.out.println("Rapid Miner initialized");

		File featureOutputFolder = new File("./xmeans_clusters/");
		if (!featureOutputFolder.exists())
			featureOutputFolder.mkdir();

		clusteringFile = new File(featureOutputFolder, "clusters.csv");
		outputClusteredTable = new File(featureOutputFolder, "clustering_table_xmeans.csv");
		outputClusterStatsTable = new File(featureOutputFolder, "cluster_stats_table_xmeans.csv");

		CSVLoader loader = new CSVLoader();
		StringBuffer sb = new StringBuffer();

		for (int i = -1; i < featureMatrix_std.length; i++) {
			for (int j = 0; j < featureMatrix_std[0].length; j++) {
				if (i == -1)
					sb.append("F" + j);
				else
					sb.append(featureMatrix_std[i][j]);
				if (j < (featureMatrix_std[0].length - 1)) {
					sb.append(",");
				} else
					sb.append("\n");
			}
		}

		InputStream tis = new ByteArrayInputStream(sb.toString().getBytes("UTF-8"));
		loader.setSource(tis);
		Instances id = loader.getDataSet();
		
		long ti = System.currentTimeMillis();
		System.out.println("XMeans: Clustering ...");
		XMeans xmeans = new XMeans();
		xmeans.setMaxIterations(maxIterations);
		xmeans.setMinNumClusters(minClusters);
		xmeans.setMaxNumClusters(maxClusters);
		xmeans.buildClusterer(id);
		System.out.println("XMEANS: ...ELAPSED CLUSTERING TIME: " + (System.currentTimeMillis() - ti));

		// do clustering
		Instances is = xmeans.getClusterCenters();
		int nClusters = is.numInstances();
		System.out.println("Estimated " + nClusters + " best clusters");
		// take results
		/*
		 * for (Instance i : is) { DenseInstance di = (DenseInstance) i; int nCluster =
		 * di.numAttributes(); for (int k = 0; k < nCluster; k++) {
		 * System.out.print(di.toString(k)+" "); }
		 * System.out.println("-------------------------------"); }
		 */
		int[] clusteringAssignments = xmeans.m_ClusterAssignments;

		String columnsNames = "id,label,cluster_id,is_an_outlier\n";
		int minpoints = minElementsInCluster;

		StringBuffer bufferRows = new StringBuffer();
		bufferRows.append(columnsNames);
		int nrows = featureMatrix_std.length;
		int ncols = featureMatrix_std[0].length;

		for (int k = 0; k < nrows; k++) {
			int cindex = clusteringAssignments[k];
			boolean isoutlier = false;
			bufferRows.append((k + 1) + ",F" + (k + 1) + "," + cindex + "," + isoutlier + "\n");
		}

		BufferedWriter bwx = new BufferedWriter(new FileWriter(clusteringFile));
		bwx.write(bufferRows.toString());
		bwx.close();
		// System.exit(0);
		/*
		 * int[] clusteringAssignments = xmeans.m_ClusterAssignments; int[] counters =
		 * new int[nClusters];
		 * 
		 * for (int cluster:clusteringAssignments){ counters[cluster]++; }
		 * 
		 * 
		 * // save the model outputClusteringModelFile = new File(featureOutputFolder,
		 * "model_clustering_xmeans.bin"); outputClusteredTable = new
		 * File(featureOutputFolder, "clustering_table_xmeans.csv");
		 * outputClusterStatsTable = new File(featureOutputFolder,
		 * "cluster_stats_table_xmeans.csv");
		 * 
		 * clusterer.save(outputClusteringModelFile);
		 */

		System.out.println("Analysing the clusters");
		List<String> clusteredFeatures = Files.readAllLines(clusteringFile.toPath());
		List<Double[]> clusteredtable = new ArrayList<Double[]>();
		HashMap<Integer, List<Double[]>> clustersWithPoints = new HashMap<Integer, List<Double[]>>();

		for (int clustidx = 1; clustidx < featureMatrix.length; clustidx++) {

			String clusteringLine = clusteredFeatures.get(clustidx);
			String clusteringLineElements[] = clusteringLine.split(",");
			int featureIdx = Integer.parseInt(clusteringLineElements[0]) - 1;
			int clusterId = Integer.parseInt(clusteringLineElements[2]);

			Double[] row = new Double[nfeatures + 1];
			for (int k = 0; k < nfeatures; k++) {
				row[k] = featureMatrix[featureIdx][k];
			}
			row[nfeatures] = (double) clusterId;

			List<Double[]> pointlist = clustersWithPoints.get(clusterId);

			if (pointlist == null)
				pointlist = new ArrayList<Double[]>();

			pointlist.add(row);
			clustersWithPoints.put(clusterId, pointlist);

			Double[] rowcomplete = new Double[allFeatureMatrix[0].length + 1];
			for (int k = 0; k < allFeatureMatrix[0].length; k++) {
				rowcomplete[k] = allFeatureMatrix[featureIdx][k];
			}
			rowcomplete[allFeatureMatrix[0].length] = (double) clusterId;

			clusteredtable.add(rowcomplete);
		}
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputClusteredTable));

		bw.write("longitude,latitude," + headerString.toString() + "clusterid\n");
		for (Double[] row : clusteredtable) {
			int c = 0;
			for (Double r : row) {
				bw.write("" + r);
				if (c < (row.length - 1))
					bw.write(",");
				else
					bw.write("\n");
				c++;
			}
		}

		bw.close();
		System.out.println("Output of clustering is in file " + clusteringFile.getAbsolutePath());

		System.out.println("Calculating clusters' stats");
		List<double[]> clusterMeans = new ArrayList<double[]>();
		for (Integer key : clustersWithPoints.keySet()) {

			List<Double[]> clustervectors = clustersWithPoints.get(key);
			int npointspercluster = clustervectors.size();
			double[] means = new double[nfeatures + 1];
			means[0] = (double) key;

			for (int i = 1; i < means.length; i++) {
				for (Double[] vector : clustervectors) {
					means[i] = means[i] + vector[i - 1];
				}
				means[i] = means[i] / (double) npointspercluster;
			}

			clusterMeans.add(means);
		}

		bw = new BufferedWriter(new FileWriter(outputClusterStatsTable));
		bw.append("clusterid," + headerString.toString().substring(0, headerString.toString().length() - 1) + "\n");

		for (double[] row : clusterMeans) {
			int c = 0;
			for (double r : row) {
				bw.write("" + r);
				if (c < (row.length - 1))
					bw.write(",");
				else
					bw.write("\n");
				c++;
			}
		}
		bw.close();
		System.out.println("Done.");

	}

	public static double percentile(double[] vector, double percentile, boolean skipzeros) {

		List<Double> latencies = Arrays.asList(ArrayUtils.toObject(vector));

		Collections.sort(latencies);

		if (skipzeros) {
			List<Double> latenciesnozero = new ArrayList<Double>();
			for (Double lat : latencies) {
				if (lat != 0d)
					latenciesnozero.add(lat);
			}
			latencies = latenciesnozero;
		}
		int index = (int) Math.ceil(percentile / 100.0 * latencies.size());
		return latencies.get(index - 1);
	}
}
