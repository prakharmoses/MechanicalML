from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import modelapi

# Defining the origins that are allowed to make requests
origins = ["http://localhost:3000"]

app = FastAPI()

# Adding the midddleware to the fast api app
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = False,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

app.include_router(modelapi.router)