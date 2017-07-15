package com.ygsoft.research;

import java.util.List;

import kr.co.shineware.nlp.komoran.core.analyzer.Komoran;
import kr.co.shineware.util.common.model.Pair;

public class HanProcessor {
	private Komoran komoran = null ;
	
	public HanProcessor() {
		komoran = new Komoran("dicdata") ;
	}
	
	public String filterNN(String src) {
		StringBuffer sb = new StringBuffer();
		
		List<List<Pair<String,String>>> tokens = this.komoran.analyze(src) ;
		
		for (List<Pair<String, String>> eojeolResult : tokens) {
			for (Pair<String, String> wordMorph : eojeolResult) {
				String snd = wordMorph.getSecond() ;
				if(snd != null && snd.startsWith("NN")) {
//					System.out.print(wordMorph.getFirst() + " ");
					sb.append(wordMorph.getFirst()).append(" ");
				}
			}
		}
		
		return sb.toString() ;
	}
	
	
	public static void main(String ... v) {
		HanProcessor hanProcessor = new HanProcessor();
		
		System.out.println("NNfiltered :" + hanProcessor.filterNN("학부모로 추정되는 맘카페 회원은 해당 어린이집 원장이 “정수기가 없어 정수되지 않은 물에 아무 건더기 없이 된장만 푼 국을 만들고 냉동김과 함께 아이들에게 제공했다”고 주장했다."));
		
	}
}
