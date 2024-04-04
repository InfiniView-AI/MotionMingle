import json
from aiohttp import web
from server import broadcast, consume
import pytest

@pytest.fixture
def cli(event_loop, aiohttp_client):
    app = web.Application()
    app.router.add_post('/broadcast', broadcast)
    app.router.add_options('/broadcast', broadcast)
    app.router.add_post("/consumer", consume)
    app.router.add_options("/consumer", consume)
    return event_loop.run_until_complete(aiohttp_client(app))

@pytest.mark.asyncio
async def test_broadcast_options(cli):
    response = await cli.options('/broadcast')
    assert response.status == 200


@pytest.mark.asyncio
async def test_broadcast_post(cli):
    sample_request_data = {
        "sdp": "my-sample-sdp",
        "type": "offer",
    }
    resp = await cli.post('/broadcast', data=json.dumps(sample_request_data))
    assert resp.status == 200
    response_data = await resp.json()
    assert "sdp" in response_data and "type" in response_data

@pytest.mark.asyncio
async def test_broadcast_empty_body_post(cli):
    empty_request_data = {}
    resp = await cli.post('/broadcast', data=json.dumps(empty_request_data))
    assert resp.status == 400

@pytest.mark.asyncio
async def test_broadcast_post_with_invalid_data_types(cli):
    invalid_data = {
        "sdp": 12345,  # Should be a string
        "type": ["offer"],  # Should be a string
    }
    resp = await cli.post('/broadcast', data=json.dumps(invalid_data))
    assert resp.status == 400

@pytest.mark.asyncio
async def test_broadcast_post_missing_fields(cli):
    incomplete_data = {
        "sdp": "my-sample-sdp",
        # Missing "type" field
    }
    resp = await cli.post('/broadcast', data=json.dumps(incomplete_data))
    assert resp.status == 400

@pytest.mark.asyncio
async def test_invalid_content_type(cli):
    headers = {"Content-Type": "text/plain"}
    sample_request_data = "This is not a valid JSON string."
    resp = await cli.post('/broadcast', headers=headers, data=sample_request_data)
    assert resp.status == 500  #Unsupported Media Type

@pytest.mark.asyncio
async def test_method_not_allowed(cli):
    resp = await cli.get('/broadcast')  # Assuming GET is not allowed
    assert resp.status == 405

@pytest.mark.asyncio
async def test_response_content_type(cli):
    sample_request_data = {
        "sdp": "my-sample-sdp",
        "type": "offer",
    }
    resp = await cli.post('/broadcast', data=json.dumps(sample_request_data))
    assert resp.status == 200
    assert resp.headers["Content-Type"] == "application/json; charset=utf-8"

@pytest.mark.asyncio
async def test_consume_options(cli):
    response = await cli.options('/consumer')
    assert response.status == 200

@pytest.mark.asyncio
async def test_consume_post(cli):
    sample_request_data = {
        "sdp": "my-sample-sdp",
        "type": "offer",
        "video_transform": "skeleton"
    }
    resp = await cli.post('/consumer', data=json.dumps(sample_request_data))
    assert resp.status == 500
    # response_data = await resp.json()
    # assert "sdp" in response_data and "type" in response_data

@pytest.mark.asyncio
async def test_consume_empty_body_post(cli):
    empty_request_data = {}
    resp = await cli.post('/consumer', data=json.dumps(empty_request_data))
    assert resp.status == 400

@pytest.mark.asyncio
async def test_consume_post_unsupported_video_transform(cli):
    invalid_data = {
        "sdp": "my-sample-sdp",
        "type": "offer",
        "video_transform": "unsupported_value"
    }
    resp = await cli.post('/consumer', data=json.dumps(invalid_data))
    assert resp.status == 400  # Assuming 400 for unsupported values, adjust as needed

