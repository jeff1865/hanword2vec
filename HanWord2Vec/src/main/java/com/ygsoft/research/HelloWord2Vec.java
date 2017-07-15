package com.ygsoft.research;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.log4j.Logger;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.UiServer;
import org.nd4j.linalg.api.ndarray.INDArray;

import kr.co.shineware.nlp.komoran.core.analyzer.Komoran;
import kr.co.shineware.util.common.model.Pair;

public class HelloWord2Vec {
	private static Logger log = Logger.getLogger(HelloWord2Vec.class);
	
	private HanProcessor hanProcessor = null ;
	
	public HelloWord2Vec() {
		this.hanProcessor = new HanProcessor();
	}
	
	public void filterNN() {
		Komoran komoran = new Komoran("dicdata") ;
		
		BufferedReader br = null;
		
		try {
			br = new BufferedReader(new FileReader("/Users/1002000/sample_han.txt"));
			
			String line = null; 
			while((line = br.readLine()) != null) {
				List<List<Pair<String,String>>> result = komoran.analyze(line);
				
				for (List<Pair<String, String>> eojeolResult : result) {
					for (Pair<String, String> wordMorph : eojeolResult) {
						String snd = wordMorph.getSecond() ;
						if(snd != null && snd.startsWith("NN")) {
							System.out.print(wordMorph.getFirst() + " ");
						}
					}
				}
				
				System.out.println();
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if(br != null) br.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
	}
	
	
	
	public void process() throws Exception {
		
		log.info("Load data....");
		SentenceIterator iter = new LineSentenceIterator(new File("/home/jeff/dev/text/sohogangho.txt"));
		iter.setPreProcessor(new SentencePreProcessor() {
		    @Override
		    public String preProcess(String sentence) {
//		    	return sentence.toLowerCase();
		    	return hanProcessor.filterNN(sentence) ;
		    }
		});
		
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());
		
		int batchSize = 1000;
		int iterations = 3;
		int layerSize = 2;

		log.info("Build model....");
		Word2Vec vec = new Word2Vec.Builder()
			.batchSize(batchSize) //# words per minibatch.
			.minWordFrequency(3) //
			.useAdaGrad(false) //
			.layerSize(layerSize) // word feature vector size
			.iterations(iterations) // # iterations to train
			.learningRate(0.025) //
			.minLearningRate(1e-3) // learning rate decays wrt # words. floor learning
			.negativeSample(10) // sample size 10 words
			.iterate(iter) //
			.tokenizerFactory(t)
			.build();
		vec.fit();
		
		// Write word vectors
		WordVectorSerializer.writeWordVectors(vec, "/home/jeff/dev/text/vector4.txt");

		log.info("Closest Words:");
		Collection<String> lst = vec.wordsNearest("소림사", 20);
		System.out.println(lst);
//		UiServer server = UiServer.getInstance();
//		System.out.println("Started on port " + server.getPort());
		
		log.info("Plot TSNE....");
		BarnesHutTsne tsne = new BarnesHutTsne.Builder()
			.setMaxIter(200)
			.stopLyingIteration(250)
			.learningRate(500)
			.useAdaGrad(false)
			.theta(0.5)
			.setMomentum(0.5)
			.normalize(true)
			.usePca(false)
			.build();
		
		vec.lookupTable().plotVocab(tsne, 20, new File("/home/jeff/dev/text/visual4.txt"));
		
	}
	
	public void visualize() {
		File model = new File("/home/jeff/dev/text/vector1.txt");
		try {
			WordVectors vec = WordVectorSerializer.loadTxtVectors(model);
			String qryWord = "검법";
			Collection<String> lst = vec.wordsNearest(qryWord, 20);
			System.out.println(qryWord + " -> " + lst);
			
//			WordVectorSerializer.loadTxt(model).getFirst().getSyn0();
			
			
			log.info("Plot TSNE....");
			BarnesHutTsne tsne = new BarnesHutTsne.Builder()
				.setMaxIter(3)
				.stopLyingIteration(250)
				.learningRate(500)
				.useAdaGrad(false)
				.theta(0.5)
				.setMomentum(0.5)
				.normalize(true)
				.usePca(false)
				.build();
			
//			UiServer server = UiServer.getInstance();
			
			vec.lookupTable().plotVocab(tsne, 1000, new File("/home/jeff/dev/text/visual6.txt"));
//			vec.lookupTable().plotVocab(tsne, 100, server.getConnectionInfo());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
	}
	
	public void visualize2() {
		File model = new File("/home/jeff/dev/text/vector1.txt");
		try {
						
			INDArray weight = WordVectorSerializer.loadTxt(model).getFirst().getSyn0();
						
			log.info("Plot TSNE....");
//			BarnesHutTsne tsne = new BarnesHutTsne.Builder()
//				.setMaxIter(200)
//				.stopLyingIteration(250)
//				.learningRate(500)
//				.useAdaGrad(false)
//				.theta(0.5)
//				.setMomentum(0.5)
//				.normalize(true)
//				.usePca(false)
//				.build();
			BarnesHutTsne tsne = new BarnesHutTsne.Builder()
	                .setMaxIter(3).theta(0.5)
	                .normalize(false)
	                .learningRate(500)
	                .useAdaGrad(false)
//	                .usePca(false)
	                .build();
			
			List<String> cacheList = new ArrayList<String>();
			tsne.plot(weight, 2, cacheList, "/home/jeff/dev/text/visual5.txt");
			
//			UiServer server = UiServer.getInstance();
//			
//			vec.lookupTable().plotVocab(tsne, 20, new File("/home/jeff/dev/text/visual4.txt"));
//			vec.lookupTable().plotVocab(20, server);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
	}
	
	public static void main(String ... v) {
		System.out.println("Start System ..");
		
		HelloWord2Vec test = new HelloWord2Vec() ;
		try {
//			test.process();
			test.visualize();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
