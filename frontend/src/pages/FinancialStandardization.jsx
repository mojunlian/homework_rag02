import React, { useState, useEffect } from 'react';
import { apiBaseUrl } from '../config/config';

const FinancialStandardization = () => {
  const [inputText, setInputText] = useState('');
  const [entityTypes, setEntityTypes] = useState([]);
  const [selectedTypes, setSelectedTypes] = useState([]);
  const [recognizedEntities, setRecognizedEntities] = useState([]);
  const [standardizedResults, setStandardizedResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiKey, setApiKey] = useState(localStorage.getItem('DEEPSEEK_API_KEY') || '');

  useEffect(() => {
    fetchEntityTypes();
  }, []);

  const fetchEntityTypes = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/financial/types`);
      const data = await response.json();
      setEntityTypes(data.types);
      setSelectedTypes(data.types);
    } catch (err) {
      console.error('Error fetching entity types:', err);
      setError('无法获取实体类型');
    }
  };

  const handleApiKeyChange = (e) => {
    const key = e.target.value;
    setApiKey(key);
    localStorage.setItem('DEEPSEEK_API_KEY', key);
  };

  const handleRecognize = async () => {
    if (!inputText.trim()) {
      setError('请输入需要分析的文本');
      return;
    }
    if (!apiKey) {
      setError('请设置 API Key');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/financial/recognize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: inputText,
          entity_types: selectedTypes,
          api_key: apiKey
        })
      });
      const data = await response.json();
      setRecognizedEntities(data.entities);
      setStandardizedResults([]); // Clear previous results
    } catch (err) {
      console.error('Error recognizing entities:', err);
      setError('实体识别失败，请检查 API Key 或网络连接');
    } finally {
      setLoading(false);
    }
  };

  const handleStandardize = async (entity, type) => {
    setLoading(true);
    try {
      const response = await fetch(`${apiBaseUrl}/financial/standardize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          entity: entity,
          entity_type: type,
          api_key: apiKey
        })
      });
      const data = await response.json();
      
      setStandardizedResults(prev => {
        const existing = prev.findIndex(r => r.original === entity);
        if (existing !== -1) {
          const updated = [...prev];
          updated[existing] = data;
          return updated;
        }
        return [...prev, data];
      });
    } catch (err) {
      console.error('Error standardizing entity:', err);
      setError(`标准化实体 ${entity} 失败`);
    } finally {
      setLoading(false);
    }
  };

  const handleStandardizeAll = async () => {
    if (recognizedEntities.length === 0) return;
    
    setLoading(true);
    setError(null);
    try {
      const results = [];
      for (const item of recognizedEntities) {
        const response = await fetch(`${apiBaseUrl}/financial/standardize`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            entity: item.entity,
            entity_type: item.type,
            api_key: apiKey
          })
        });
        const data = await response.json();
        results.push(data);
      }
      setStandardizedResults(results);
    } catch (err) {
      console.error('Error standardizing all entities:', err);
      setError('部分实体标准化失败');
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
        <h1 className="text-3xl font-bold text-gray-800">金融术语标准化 📚</h1>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Left Column: Input and Configuration */}
        <div className="col-span-12 lg:col-span-8 space-y-6">
          <div className="bg-white p-6 rounded-xl shadow-md border border-gray-100">
            <h2 className="text-xl font-semibold mb-4 text-gray-700 flex items-center">
              输入金融文本
            </h2>
            <textarea
              className="w-full h-48 p-4 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-all"
              placeholder="请输入需要标准化的金融文本内容..."
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
              onClick={handleRecognize}
              disabled={loading || !inputText}
              className={`mt-6 w-full py-3 px-6 rounded-lg font-semibold text-white transition-all shadow-md ${
                loading || !inputText ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 active:transform active:scale-95'
              }`}
            >
              {loading ? '正在处理...' : '识别金融实体'}
            </button>
          </div>

          {/* Recognized Entities List */}
          {recognizedEntities.length > 0 && (
            <div className="bg-white p-6 rounded-xl shadow-md border border-gray-100 animate-fadeIn">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold text-gray-700">识别到的实体</h2>
                <button
                  onClick={handleStandardizeAll}
                  disabled={loading}
                  className="text-blue-600 hover:text-blue-800 font-medium flex items-center"
                >
                  全部标准化
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-1" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">实体文本</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">实体类型</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">操作</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {recognizedEntities.map((item, index) => (
                      <tr key={index}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{item.entity}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                            {item.type}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <button
                            onClick={() => handleStandardize(item.entity, item.type)}
                            className="text-indigo-600 hover:text-indigo-900"
                          >
                            标准化
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Configuration and Results */}
        <div className="col-span-12 lg:col-span-4 space-y-6">
          <div className="bg-white p-6 rounded-xl shadow-md border border-gray-100">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">配置术语类型</h2>
            <div className="grid grid-cols-2 lg:grid-cols-1 gap-2">
              {entityTypes.map((type) => (
                <label key={type} className="flex items-center p-2 hover:bg-gray-50 rounded transition-colors cursor-pointer">
                  <input
                    type="checkbox"
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    checked={selectedTypes.includes(type)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedTypes([...selectedTypes, type]);
                      } else {
                        setSelectedTypes(selectedTypes.filter(t => t !== type));
                      }
                    }}
                  />
                  <span className="ml-3 text-sm text-gray-600">{type}</span>
                </label>
              ))}
            </div>
          </div>

          {standardizedResults.length > 0 && (
            <div className="bg-white p-6 rounded-xl shadow-md border border-gray-100 animate-fadeIn">
              <h2 className="text-xl font-semibold mb-4 text-gray-700">标准化结果</h2>
              <div className="space-y-4">
                {standardizedResults.map((result, index) => (
                  <div key={index} className="p-4 bg-gray-50 rounded-lg border-l-4 border-green-500">
                    <div className="flex justify-between items-start mb-2">
                      <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">{result.original}</span>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-green-500" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="text-lg font-bold text-gray-800 mb-1">{result.standardized}</div>
                    <p className="text-sm text-gray-600 italic">{result.explanation}</p>
                  </div>
                ))}
              </div>
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
