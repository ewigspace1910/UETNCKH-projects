from fastapi import APIRouter
from fastapi.responses import JSONResponse
from db_integration import query2db
router = APIRouter(
    prefix="/dba",
    tags=['[dev-branch] DB adapter']
)

@router.get("/query", summary="execute SQL query in our DN")
async def execute_query(query:str, db:str='palse'):
    df = None
    try:
        df = query2db(db, query)
        datetime_col_names = df.select_dtypes(include=['datetime']).columns.tolist()
        for col in datetime_col_names:
            df[col] = df[col].astype(str)
            
        return JSONResponse({
            'complete':True,
            'body': df.to_dict()
        })
    except Exception as e:
        print(str(e))
        return JSONResponse({
            'complete':False,
            'body': None
        })