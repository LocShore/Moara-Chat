from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from MoaraIA import MoaraIA

app = FastAPI()
moara = MoaraIA()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Local do servidor que vai usar
    allow_methods=["POST"], # metodo (get: apenas pega informação, post: tras e leva informação)
    allow_headers=["Content-Type"], # header do negocio (tipo de classe que define o tipo)
)

class PerguntaRequest(BaseModel):
    user_prompt: str

@app.post("/iaOutGame")
async def out_ask(body: PerguntaRequest):

    resposta = moara.out_game_response(body.user_prompt)

    return {"retorno": resposta}

@app.post("/iaInGame")
async def in_ask(body: PerguntaRequest):

    resposta = moara.in_game_response(body.user_prompt)

    return {"retorno": resposta}

