exports.handler = async (event) => {
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Content-Type': 'application/json',
  };

  if (event.httpMethod === 'OPTIONS') return { statusCode: 200, headers, body: '' };
  if (event.httpMethod !== 'POST') return { statusCode: 405, headers, body: JSON.stringify({ error: 'Method Not Allowed' }) };

  try {
    const { image } = JSON.parse(event.body || '{}');
    if (!image) return { statusCode: 400, headers, body: JSON.stringify({ error: '이미지 없음' }) };

    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) return { statusCode: 500, headers, body: JSON.stringify({ error: 'OPENAI_API_KEY 미설정' }) };

    const res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
      body: JSON.stringify({
        model: 'gpt-4o',
        max_tokens: 800,
        messages: [{
          role: 'user',
          content: [
            { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${image}`, detail: 'high' } },
            { type: 'text', text: `커피 원두 분쇄도를 분석하세요. 흰색 바탕 위 분쇄 원두 사진입니다.\n커피 원두가 아니면 isCoffee:false.\n매커포트 최적: 300~500μm.\nJSON만 반환:\n{"level":"극세분/세분/중간/굵게/아주굵게","levelKor":"한글명","percent":숫자,"particleSize":숫자,"uniformity":숫자,"mokaFit":"최적/약간 고움/많이 고움/약간 굵음/많이 굵음","bestBrew":"추출방식","advice":"2문장","isCoffee":true}` }
          ]
        }]
      })
    });

    if (!res.ok) {
      const err = await res.text();
      if (res.status === 401) return { statusCode: 401, headers, body: JSON.stringify({ error: 'API 키 오류' }) };
      if (res.status === 429) return { statusCode: 429, headers, body: JSON.stringify({ error: '한도 초과' }) };
      return { statusCode: res.status, headers, body: JSON.stringify({ error: err.slice(0, 200) }) };
    }

    const data = await res.json();
    const text = data.choices?.[0]?.message?.content || '';
    if (!text) return { statusCode: 500, headers, body: JSON.stringify({ error: '응답 없음' }) };

    let result;
    try { result = JSON.parse(text.replace(/```json|```/g, '').trim()); }
    catch { const m = text.match(/\{[\s\S]*\}/); if (!m) return { statusCode: 500, headers, body: JSON.stringify({ error: '파싱 실패' }) }; result = JSON.parse(m[0]); }

    return { statusCode: 200, headers, body: JSON.stringify({ result }) };

  } catch(err) {
    return { statusCode: 500, headers, body: JSON.stringify({ error: err.message || '서버 오류' }) };
  }
};
