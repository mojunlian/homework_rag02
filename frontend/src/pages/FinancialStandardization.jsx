import React, { useState } from 'react';
import { apiBaseUrl } from '../config/config';
import ReactMarkdown from 'react-markdown';

const FinancialStandardization = () => {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiKey, setApiKey] = useState(localStorage.getItem('DEEPSEEK_API_KEY') || '');

  const handleApiKeyChange = (e) => {
    const key = e.target.value;
    setApiKey(key);
    localStorage.setItem('DEEPSEEK_API_KEY', key);
  };

  const handleExplain = async () => {
    if (!inputText.trim()) {
      setError('请输入需要查询的金融术语或文本');
      return;
    }
    if (!apiKey) {
      setError('请设置 API Key');
      return;
    }

    setLoading(true);
    setError(null);
    setResult({ candidates: [], explanation: '' });

    try {
      const response = await fetch(`${apiBaseUrl}/financial/explain`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: inputText,
          api_key: apiKey
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedExplanation = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        
        // 处理候选词数据
        if (chunk.startsWith('DATA: ')) {
          try {
            const dataStr = chunk.substring(6).split('\n\n')[0];
            const data = JSON.parse(dataStr);
            setResult(prev => ({ ...prev, candidates: data.candidates }));
          } catch (e) {
            console.error('Error parsing candidate data:', e);
          }
        } else if (chunk.startsWith('ERROR: ')) {
          setError(chunk.substring(7));
          break;
        } else {
          // 处理流式文本
          accumulatedExplanation += chunk;
          setResult(prev => ({ ...prev, explanation: accumulatedExplanation }));
        }
      }
    } catch (err) {
      console.error('Error explaining term:', err);
      setError('处理失败，请检查 API Key 或网络连接');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6">
      <div className="flex items-center mb-8">
        <div className="bg-blue-600 p-3 rounded-lg mr-4 text-white shadow-lg">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <h1 className="text-3xl font-bold text-gray-800">金融术语助手 📚</h1>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Left Column: Input */}
        <div className="col-span-12 lg:col-span-5 space-y-6">
          <div className="bg-white p-6 rounded-xl shadow-md border border-gray-100">
            <h2 className="text-xl font-semibold mb-4 text-gray-700 flex items-center">
              查询输入
            </h2>
            <textarea
              className="w-full h-32 p-4 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-all"
              placeholder="请输入金融术语或相关文本..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />
            
            <div className="mt-4">
              <label className="block text-sm font-medium text-gray-600 mb-2">DeepSeek API Key</label>
              <input
                type="password"
                className="w-full p-2 border border-gray-200 rounded-md"
                placeholder="在此输入您的 API Key"
                value={apiKey}
                onChange={handleApiKeyChange}
              />
            </div>

            <button
              onClick={handleExplain}
              disabled={loading || !inputText}
              className={`mt-6 w-full py-3 px-6 rounded-lg font-semibold text-white transition-all shadow-md ${
                loading || !inputText ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 active:transform active:scale-95'
              }`}
            >
              {loading ? '正在分析...' : '搜索并解释'}
            </button>
          </div>
          
          {/* Candidates List */}
          {result && result.candidates && result.candidates.length > 0 && (
             <div className="bg-white p-6 rounded-xl shadow-md border border-gray-100 animate-fadeIn">
               <h3 className="text-lg font-semibold mb-3 text-gray-700">相关标准术语</h3>
               <div className="space-y-2">
                 {result.candidates.map((cand, idx) => (
                   <div key={idx} className="p-3 bg-gray-50 rounded border border-gray-100 flex justify-between items-center">
                     <div>
                       <span className="font-medium text-gray-800">{cand.term}</span>
                       <span className="ml-2 text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded-full">{cand.category}</span>
                     </div>
                     <span className="text-xs text-gray-500">相似度: {cand.distance.toFixed(4)}</span>
                   </div>
                 ))}
               </div>
             </div>
          )}
        </div>

        {/* Right Column: Result */}
         <div className="col-span-12 lg:col-span-7">
           {result ? (
             <div className="bg-white p-6 rounded-xl shadow-md border border-gray-100 h-full animate-fadeIn">
               <h2 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">解释与分析</h2>
               {result.explanation ? (
                 <div className="prose prose-blue max-w-none">
                   <ReactMarkdown>{result.explanation}</ReactMarkdown>
                 </div>
               ) : (
                 <div className="text-gray-500 italic">未收到解释内容</div>
               )}
               {/* Fallback for debugging */}
               {/* <pre className="mt-4 p-2 bg-gray-100 text-xs overflow-auto max-h-40">{JSON.stringify(result, null, 2)}</pre> */}
             </div>
           ) : (
            <div className="bg-gray-50 p-8 rounded-xl border border-gray-200 h-full flex flex-col items-center justify-center text-gray-400">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mb-4 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p>在左侧输入内容并点击按钮开始分析</p>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="fixed bottom-8 right-8 bg-red-100 border-l-4 border-red-500 text-red-700 p-4 shadow-xl animate-bounce" role="alert">
          <p className="font-bold">提示</p>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
};

export default FinancialStandardization;
