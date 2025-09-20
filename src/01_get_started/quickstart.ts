import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

// LLM Model
const chatModel = new ChatOpenAI({
  model: "gpt-4o",
});

// Prompt
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a world class technical documentation writer."],
  ["user", "{input}"],
]);

// Parser
const outputParser = new StringOutputParser();

// Chaining

const chain = prompt.pipe(chatModel).pipe(outputParser);

// invoke
console.log(
  await chain.invoke({
    input: "langsmith가 뭐야?",
  })
);

// streaming
const chucks = [];
for await (const chuck of await chain.stream({ input: "langsmith가 뭐야?" })) {
  chucks.push(chuck);
  console.log(`${chuck}\n`);
}

console.log(chucks.join(""));
