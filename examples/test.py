import asyncio


async def async_foo():
    print("async_foo started")
    await asyncio.sleep(1)
    print("async_foo done")


async def main():
    for _ in range(2):
        asyncio.ensure_future(async_foo())  # fire and forget async_foo()

    # btw, you can also create tasks inside non-async funcs

    print('Do some actions 1')
    await asyncio.sleep(1)
    print('Do some actions 2')
    await asyncio.sleep(1)
    print('Do some actions 3')


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())