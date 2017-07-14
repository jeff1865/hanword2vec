package com.ygsoft.research;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

import org.apache.log4j.Logger;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.UiServer;

import kr.co.shineware.nlp.komoran.core.analyzer.Komoran;
import kr.co.shineware.util.common.model.Pair;

public class HelloWord2Vec {
	private static Logger log = Logger.getLogger(HelloWord2Vec.class);
	
	private HanProcessor hanProcessor = null ;
	
	public HelloWord2Vec() {
		this.hanProcessor = new HanProcessor();
	}
	
	public void filterNN() {
		Komoran komoran = new Komoran("/Users/1002000/dev/myworks/crawl/crawlCore/dicdata") ;
		
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
		SentenceIterator iter = new LineSentenceIterator(new File("/Users/1002000/temp_han/han_sample.txt"));
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
		int layerSize = 150;

		log.info("Build model....");
		Word2Vec vec = new Word2Vec.Builder()
			.batchSize(batchSize) //# words per minibatch.
			.minWordFrequency(5) //
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
		WordVectorSerializer.writeWordVectors(vec, "/Users/1002000/temp_han/vector4.txt");

		log.info("Closest Words:");
		Collection<String> lst = vec.wordsNearest("노인", 20);
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
		
		vec.lookupTable().plotVocab(tsne, 100, new File("/Users/1002000/temp_han/visual4.txt"));
		
	}
	
	public static void main(String ... v) {
		System.out.println("Start System ..");
		
		HelloWord2Vec test = new HelloWord2Vec() ;
		try {
			test.process();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
