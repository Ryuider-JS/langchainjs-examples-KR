import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

// LLM Model
const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
});

// Document Loader
const loader = new CheerioWebBaseLoader(
  "https://docs.smith.langchain.com/user_guide"
);

/**
 * load 메서드 - 데이터 가져오기 -> 전처리 -> 반환
 * scrape 메서드 - 데이터 가져오기
 */
const docs = await loader.load();
// const scrape = await loader.scrape();

// console.log(docs.length);
// console.log(docs[0]);
// console.log(docs[0].pageContent.length);

/**
 * ### RecursiveCharacterTextSplitterParams
 * chuckSize(number) : 분할될 각 조각(chuck)의 최대크기를 지정
 * chuckOverlap(number) : 잘린 조각(chuck)들 사이에 겹치는 부분의 크기 지정
 * separators(string[]) : chucking할 때 사용할 구분자(separator)들의 리스트
 * keepSeparator(boolean - default false) : chucking할 때 사용한 구분자(separator)를 chuck에 남겨둘 지 여부
 */
// TextSplitter
const splitter = new RecursiveCharacterTextSplitter();

const splitDocs = await splitter.splitDocuments(docs);

// console.log(splitDocs);
// console.log(splitDocs.length);
// console.log(splitDocs[0].pageContent.length);

// Embedding Model
const embeddings = new OpenAIEmbeddings();

// VectorStore
const vectorstore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

const prompt =
  ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

// <context>
// {context}
// </context>

// Question: {input}`);

const documentChain = await createStuffDocumentsChain({
  llm,
  prompt,
});

// console.log(documentChain);

console.log(
  await documentChain.invoke({
    input: "LangSmith가 뭐야?",
    context: [
      new Document({
        pageContent:
          "LangSmith is a platform for building production-grade LLM applications.",
      }),
    ],
  })
);

// const retriever = vectorstore.asRetriever();

// const retrievalChain = await createRetrievalChain({
//   combineDocsChain: documentChain,
//   retriever,
// });

// console.log(
//   await retrievalChain.invoke({
//     input: "what is LangSmith?",
//   })
// );

// const historyAwarePrompt = ChatPromptTemplate.fromMessages([
//   new MessagesPlaceholder("chat_history"),
//   ["user", "{input}"],
//   [
//     "user",
//     "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
//   ],
// ]);

// const historyAwareRetrieverChain = await createHistoryAwareRetriever({
//   llm: chatModel,
//   retriever,
//   rephrasePrompt: historyAwarePrompt,
// });

// const chatHistory = [
//   new HumanMessage("Can LangSmith help test my LLM applications?"),
//   new AIMessage("Yes!"),
// ];

// console.log(
//   await historyAwareRetrieverChain.invoke({
//     chat_history: chatHistory,
//     input: "Tell me how!",
//   })
// );

// const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
//   [
//     "system",
//     "Answer the user's questions based on the below context:\n\n{context}",
//   ],
//   new MessagesPlaceholder("chat_history"),
//   ["user", "{input}"],
// ]);

// const historyAwareCombineDocsChain = await createStuffDocumentsChain({
//   llm: chatModel,
//   prompt: historyAwareRetrievalPrompt,
// });

// const conversationalRetrievalChain = await createRetrievalChain({
//   retriever: historyAwareRetrieverChain,
//   combineDocsChain: historyAwareCombineDocsChain,
// });

// const result2 = await conversationalRetrievalChain.invoke({
//   chat_history: [
//     new HumanMessage("Can LangSmith help test my LLM applications?"),
//     new AIMessage("Yes!"),
//   ],
//   input: "tell me how",
// });

// console.log(result2.answer);
