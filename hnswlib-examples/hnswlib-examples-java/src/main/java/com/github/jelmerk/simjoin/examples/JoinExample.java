package com.github.jelmerk.simjoin.examples;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import com.github.jelmerk.knn.DistanceFunctions;
import com.github.jelmerk.knn.SearchResult;
import com.github.jelmerk.knn.hnsw.HnswIndex;

/**
 * Example application that downloads the english fast-text word vectors,
 * inserts them into an hnsw index and lets you query them.
 */
public class JoinExample
{

	//private static final String WORDS_FILE_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz";

	//private static final Path TMP_PATH = Paths.get(System.getProperty("java.io.tmpdir"));
	private static final Path TMP_PATH = Paths.get("/media/dados/projetos/dataset/vetores/");

	//private DistanceFunction<Vector, Double> distanceFunctions;

	public static void main(String[] args) throws Exception
	{

		// Load File and create index

		long start = System.currentTimeMillis();

		Path file = TMP_PATH.resolve("vetores2500_multi-qa-MiniLM-L6-cos-v1.csv");
		//Path file = TMP_PATH.resolve("vetoresDblp100k_semMudarIntervalo_multi-qa-MiniLM-L6-cos-v1.csv");

		if (!Files.exists(file))
		{
			System.out.printf("Input file dos not exists!");
			System.exit(0);
		}

		List<Vector> vectors = loadVectorsFromCSV(file);

		// Get number of dimensions
		int dimension = vectors.get(0).dimensions();

		//Params
		int M = 32;
		int efConstruction = 16;
		int ef = 16;
		double threshold = 0.9;

		// FloatInnerProduct
		/*HnswIndex<String, float[], Word, Float> hnswIndex = HnswIndex
				.newBuilder(dimension, DistanceFunctions.FLOAT_INNER_PRODUCT, words.size()).withM(M).withEf(ef)
				.withEfConstruction(efConstruction).build();*/

		//DooubleCosine
		HnswIndex<Integer, double[], Vector, Double> hnswIndex = HnswIndex
				.newBuilder(dimension, DistanceFunctions.DOUBLE_COSINE_DISTANCE, vectors.size()).withM(M).withEf(ef)
				.withEfConstruction(efConstruction).build();

		/*hnswIndex.addAll(words,
				(workDone, max) -> System.out.printf("Added %d out of %d vectors to the index.%n", workDone, max));*/

		hnswIndex.addAll(vectors);

		long end = System.currentTimeMillis();

		long timeIndex = end - start;

		//Index<String, float[], Word, Float> groundTruthIndex = hnswIndex.asExactIndex();

		/*
		
		///////////////////////////////////
		///////////////////////////////////
		/////// Example of the unique query
		
		start = System.currentTimeMillis();
		
		//vector to query
		int vectorQuery = 85;
		
		float[] query = words.get(vectorQuery).vector();
		
		int k = 5;
		
		float threshold = 0.2f; //TODO o algortimo espera a distancia 1 - ths
		
		
		//List<SearchResult<Word, Float>> approximateResults = hnswIndex.findNeighbors(query, k);
		//List<SearchResult<Word, Float>> approximateResults = hnswIndex.findNearest(query, k);
		
		List<SearchResult<Word, Float>> approximateResults = hnswIndex.rangeSearch(query, k, threshold);
		
		//List<SearchResult<Word, Float>> groundTruthResults = groundTruthIndex.findNeighbors(query, k);
		//List<SearchResult<Word, Float>> groundTruthResults = groundTruthIndex.findNearest(query, k);
		
		end = System.currentTimeMillis();
		
		long timeJoin = end - start;
		
		//System.out.println("\nMost similar vectors found using HNSW index:");
		//System.out.println("ID Vetor Consulta, ID Vizinho ( Foreing Key Vetor Consulta, ForeignKey Vizinho)");
		for (SearchResult<Word, Float> result : approximateResults)
		{
			System.out.printf("%s, %s (%s, %s):[%.6f]%n", vectorQuery + 1, result.item().id(),
					words.get(vectorQuery).foreingKey(), result.item().foreingKey(), 1 - result.distance());
		}
		
		System.out.println("----- Run Information ----");
		System.out.println("Pairs Total: " + approximateResults.size());
		System.out.printf("Total time: %dms%n", timeIndex + timeJoin);
		System.out.printf("Create index time: %dms%n", timeIndex);
		System.out.printf("Join time: %dms%n", timeJoin);
		
		//System.out.println("\nMost similar vectors found using exact index:");
		
		/*for (SearchResult<Word, Float> result : groundTruthResults)
		{
			System.out.printf("%s, %s (%s, %s):[%.6f]%n", vectorQuery+1, result.item().id(), words.get(vectorQuery).foreingKey(),result.item().foreingKey(), 1-result.distance());
		}*/

		//int correct = groundTruthResults.stream().mapToInt(r -> approximateResults.contains(r) ? 1 : 0).sum();

		//System.out.printf("%nAccuracy : %.4f%n%n", correct / (double) groundTruthResults.size());

		//////////////////////////////////////////////////
		//////////////////////////////////////////////////
		//////////////////////////////////////////////////

		//*//

		/////////////////////////////////////////////////
		/////////////////////////////////////////////////
		// Example Similarity Join

		start = System.currentTimeMillis();

		////// with top-k to threshold

		/*
		int k = 10;
		double threshold = 0.9;
		List<String> results = new ArrayList<>();
		for (Vector w : words)
		{
			double[] q = w.vector();
			//int id = Integer.parseInt(w.id());
			int id = w.id();
			List<SearchResult<Vector, Double>> annResults = hnswIndex.findNearest(q, k);
		
			for (SearchResult<Vector, Double> result : annResults)
			{
		
				//hnswIndex.remove(w.id(),1);
		
				//int resultID = Integer.parseInt(result.item().id());
				int resultID = result.item().id();
				
				if (1 - result.distance() > threshold && resultID > id) //not include repeated pairs
				{
					double distance = 1 - result.distance();
					//id = id - 1;
					String r = (id + ", " + result.item().id() + " (" + words.get(id - 1).foreingKey() + ", "
							+ result.item().foreingKey() + ") : [" + distance + "]");
					//System.out.printf("%s, %s (%s, %s):[%.6f]%n", id-1, result.item().id(), words.get(id-1).foreingKey(),result.item().foreingKey(), 1-result.distance());
					results.add(r);
		
				}
			}
		}
		
		//////////////////
		//////////////////
		
		//*/

		//*
		// with rangeSearch

		//adjust to distance
		double ths = 1 - threshold;
		List<String> results = new ArrayList<>();

		//List<Vector> filter = new ArrayList<>();
		//int count = 0;
		//DistanceFunction<double[], Double> distanceFunction = DistanceFunctions.DOUBLE_COSINE_DISTANCE;
		//double d;

		for (Vector v : vectors)
		{
			double[] q = v.vector();
			int id = v.id();

			// Filter vectors
			//d = distanceFilter.distance(q, q);

			//if (!filter.isEmpty())
			//{
				//d = distanceFunction.distance(q, filter.get(filter.size()-1).vector());

			//}

			// use rangeSearch or rangeSearch2
			List<SearchResult<Vector, Double>> annResults = hnswIndex.rangeSearch(q, ths);

			/*// check if it didn't return results and use vector as filter
			if (annResults.size() < 2)
			{
				//System.out.println("teste");
				filter.add(v);
			
			}*/

			for (SearchResult<Vector, Double> result : annResults)
			{

				//hnswIndex.remove(w.id(),1);

				//int resultID = Integer.parseInt(result.item().id());
				int resultID = result.item().id();

				//	if (1 - result.distance() > threshold && resultID > id) //not include repeated pairs
				//if (result.distance() < ths && resultID > id) //not include repeated pairs
				if (resultID > id) //not include repeated pairs
				{
					double distance = 1 - result.distance();
					//id = id - 1;
					String r = (id + ", " + result.item().id() + " (" + vectors.get(id - 1).foreingKey() + ", "
							+ result.item().foreingKey() + ") : [" + distance + "]");
					//System.out.printf("%s, %s (%s, %s):[%.6f]%n", id-1, result.item().id(), words.get(id-1).foreingKey(),result.item().foreingKey(), 1-result.distance());
					results.add(r);

				}
			}
		}

		//*/

		end = System.currentTimeMillis();

		long timeJoin = end - start;

		// Print results
		for (String r : results)
		{
			System.out.println(r);
		}
		System.out.println("----- Run Information ----");
		//System.out.println("Filtered vectors: " + count);
		System.out.println("Pairs Total: " + results.size());
		System.out.printf("Total time: %dms%n", timeIndex + timeJoin);
		System.out.printf("Create index time: %dms%n", timeIndex);
		System.out.printf("Join time: %dms%n", timeJoin);
		System.out.println("Params: Index=HNSW, M=" + M + ", EfConstruction=" + efConstruction + ", EfSearch=" + ef);
		System.out.println("Threshold: " + threshold);
		System.out.println("File: " + file);
		System.out.println("Total vectors and dimensions: (" + vectors.size() + "," + dimension + ")");

		//////////////////////////////////////////////////
		//////////////////////////////////////////////////
		//////////////////////////////////////////////////

	}

	private static List<Vector> loadVectorsFromCSV(Path path) throws IOException
	{
		//System.out.printf("Loading vectors from %s%n", path);

		try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(path.toString()))))
		{

			int count = 1;
			String line = "";

			List<Vector> vectors = new ArrayList<>();

			while ((line = reader.readLine()) != null)
			{
				String[] tokens = line.split(",");
				//String id = String.valueOf(count);
				int id = Integer.valueOf(count);

				//String foreingKey = tokens[0];
				int foreingKey = Integer.valueOf(tokens[0]);

				double[] vector = new double[tokens.length - 1];
				for (int i = 1; i < tokens.length - 1; i++)
				{
					vector[i] = Float.parseFloat(tokens[i]);
				}

				Vector w = new Vector(id, foreingKey, vector);
				vectors.add(w);
				count++;
			}
			return vectors;
		}

	}

}
