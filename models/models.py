from pydantic import BaseModel


class OnlineFraud(BaseModel):
    """
    Represents Online transaction atributes to predict if its fraud or not

    Atributes:
        type
        amount
        oldbalanceOrig
        newbalanceOrig
    """

    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
